from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch.utils.data as Data
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
import tqdm
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split

from parse import *
from metric import *
from utils import *
from model import *
from dataset import *

get_data_map = {'yelp': generate_yelp_data, 'clickbait': generate_clickbait_data, 'imdb': generate_imdb_data,
                'hate': generate_hate_data}
get_epoch_map = {'yelp': 50, 'clickbait': 100, 'imdb': 50, 'hate': 100}
get_dis_map = {'yelp': 1., 'clickbait': 1., 'imdb': 0.1, 'hate': 1.}

args = parse_args(None)
device = torch.device("cuda", args.device)
setup_seed()

text, label = get_data_map[args.dataset]()
split = int(len(text) * 0.8)
train_titles, train_labels = text[:split], label[:split]
test_titles, test_labels = text[split:], label[split:]

assert len(train_titles) == len(train_labels)
assert len(test_titles) == len(test_labels)

train_dataset = My_dataset(train_titles, train_labels)
test_dataset = My_dataset(test_titles, test_labels)

model = Bert_stack(args).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_mse = torch.nn.MSELoss(reduction='mean')

best_acc = 0.
best_mse = 100.

weights = torch.load(f"save/{args.dataset}_bert.pt", map_location=device)
model.load_state_dict(weights, strict=False)

model_path = f"save/{args.dataset}_stack"
if (not os.path.exists(model_path)):
    os.makedirs(model_path)
nb_epochs = get_epoch_map[args.dataset]
for epoch in range(0, nb_epochs):
    model.train()
    optimizer.zero_grad()

    loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    pbar = tqdm.tqdm(loader, "train", total=len(test_dataset) // args.batch_size)
    acc_list = []
    iou_list = []
    comp_list = []
    suff_list = []
    length_list = []
    mse_list = []
    select_list = []
    trans_list = []
    for (batch_x, batch_y) in pbar:
        pred, pred1, mask, lengths, logits1, logits2, mask1, mask2 = model.forward_dis1(batch_x)
        pred_true, _, detail_output1 = model.forward_single(batch_x)
        batch_y = batch_y.to(device).float()

        ce_loss = loss_mse(pred, pred_true) - loss_mse(pred1, pred_true)
        prob_loss = torch.mean(-F.logsigmoid(mask1 - mask2))
        dis_loss = generate_sim(F.softmax(logits1, dim=-1), F.softmax(logits2, dim=-1))
        selection_loss, transitions_loss, g_loss, valid_mask, _ = generator_loss(mask, lengths, device, args)
        loss = ce_loss + prob_loss + dis_loss * get_dis_map[args.dataset]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_comp = model.forward_mask(batch_x, torch.lt(valid_mask, 0.5)).detach().cpu().numpy()
        pred_suff = model.forward_mask(batch_x, torch.gt(valid_mask, 0.5)).detach().cpu().numpy()

        batch_y = batch_y.cpu().numpy()
        pred = pred.detach().cpu().numpy()
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(batch_y, axis=1))
        acc_list.append(acc)
        iou_list += IOU_list(model.tokenizer, batch_x, batch_x, valid_mask.detach().cpu().numpy())

        comp_list += difference_list(pred, pred_comp, batch_y)
        suff_list += difference_list(pred, pred_suff, batch_y)
        mse_list.append(loss.cpu().item())
        select_list.append(selection_loss.sum().detach().cpu().numpy())
        trans_list.append(transitions_loss.sum().detach().cpu().numpy())

        length_list.append(np.mean(np.sum((valid_mask > 0.5).detach().cpu().numpy(), axis=1)))

        pbar.set_postfix(mse_loss=loss.cpu().item(), acc=acc,
                         selection="%.4f" % selection_loss.sum().detach().cpu().numpy(),
                         transitions="%.4f" % g_loss.sum().detach().cpu().numpy(),
                         IOU=np.mean(iou_list), comp="%.4f" % np.mean(comp_list), suff="%.4f" % np.mean(suff_list),
                         length=np.mean(length_list))

    print(
        "after epoch %d, train acc=%f train IOU=%f train Comp=%f train Suff=%f train loss=%f train length=%f train select=%f train trans=%f" % (
            epoch, np.mean(acc_list), np.mean(iou_list), np.mean(comp_list), np.mean(suff_list), np.mean(mse_list),
            np.mean(length_list), np.mean(select_list), np.mean(trans_list)))

    if (epoch % 5 == 0 or epoch == nb_epochs - 1):
        model.eval()
        loader = Data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        pbar = tqdm.tqdm(loader, "test", total=len(test_dataset) // args.test_batch_size)
        acc_list = []
        iou_list = []
        mse_list = []
        pred_list = []
        label_list = []
        comp_list = []
        suff_list = []
        length_list = []
        for (batch_x, batch_y) in pbar:
            pred, mask, lengths, _ = model(batch_x)
            batch_y = batch_y.to(device)

            loss = loss_mse(pred, batch_y.float()).sum()
            mse_list.append(loss.cpu().item())

            batch_y = batch_y.cpu().numpy()
            pred = pred.detach().cpu().numpy()
            pred_list.append(pred)
            label_list.append(batch_y)
            selection_loss, transitions_loss, g_loss, valid_mask, _ = generator_loss(mask, lengths, device, args)

            pred_comp = model.forward_mask(batch_x, torch.lt(valid_mask, 0.5)).detach().cpu().numpy()
            pred_suff = model.forward_mask(batch_x, torch.gt(valid_mask, 0.5)).detach().cpu().numpy()

            iou_list += IOU_list(model.tokenizer, batch_x, batch_x, valid_mask.detach().cpu().numpy())

            comp_list += difference_list(pred, pred_comp, batch_y)
            suff_list += difference_list(pred, pred_suff, batch_y)
            length_list.append(np.mean(np.sum((valid_mask > 0.5).detach().cpu().numpy(), axis=1)))

        pred = np.concatenate(pred_list, axis=0)
        batch_y = np.concatenate(label_list, axis=0)
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(batch_y, axis=1))
        acc_list.append(acc)
        torch.save(model.state_dict(),
                   f"{model_path}/{args.dataset}_stack.pt")
        print("after epoch %d, test acc=%f test IOU=%f test Comp=%f test Suff=%f test loss=%f test length=%f" % (
            epoch, np.mean(acc_list), np.mean(iou_list), np.mean(comp_list), np.mean(suff_list), np.mean(mse_list),
            np.mean(length_list)))
