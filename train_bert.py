from transformers import BertModel, BertTokenizer
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
from torch import nn
import tqdm
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split

from parse import *
from metric import *
from utils import *
from dataset import *
from model import *

args = parse_args(None)

device = torch.device("cuda", args.device)
setup_seed()  # set random seed

# get dataset
get_data_map = {'yelp': generate_yelp_data, 'clickbait': generate_clickbait_data, 'imdb': generate_imdb_data,
                'hate': generate_hate_data}
text, label = get_data_map[args.dataset]()
split = int(len(text) * 0.8)
train_titles, train_labels = text[:split], label[:split]
test_titles, test_labels = text[split:], label[split:]

train_dataset = My_dataset(train_titles, train_labels)
test_dataset = My_dataset(test_titles, test_labels)

model = Bert(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_ce = torch.nn.BCELoss()

best_acc = 0.
for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    loader = Data.DataLoader(dataset=train_dataset, batch_size=8, num_workers=2, shuffle=True)
    pbar = tqdm.tqdm(loader, "train", total=len(train_dataset) // 8)
    acc_list = []
    for (batch_x, batch_y) in pbar:
        pred, detail_output, lengths = model(batch_x)
        batch_y = batch_y.to(device)

        ce_loss = loss_ce(pred[:, 1], batch_y.float()[:, 1])
        loss = ce_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_y = batch_y.cpu().numpy()
        pred = pred.detach().cpu().numpy()
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(batch_y, axis=1))
        acc_list.append(acc)
        pbar.set_postfix(mse_loss=ce_loss.cpu().item(), acc=acc)
    print("after epoch %d, train acc=%f" % (epoch, np.mean(acc_list)))

    model.eval()
    loader = Data.DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=2)
    pbar = tqdm.tqdm(loader, "test", total=len(test_dataset) // 2)
    acc_list = []
    pred_list = []
    label_list = []
    for (batch_x, batch_y) in pbar:
        pred, detail_output, lengths = model(batch_x)
        batch_y = batch_y.to(device)

        batch_y = batch_y.cpu().numpy()
        pred = pred.detach().cpu().numpy()
        pred_list.append(pred)
        label_list.append(batch_y)
    pred = np.concatenate(pred_list, axis=0)
    batch_y = np.concatenate(label_list, axis=0)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(batch_y, axis=1))
    acc_list.append(acc)
    if (np.mean(acc_list) > best_acc):
        best_acc = np.mean(acc_list)
        torch.save(model.state_dict(), f"save/{args.dataset}_bert.pt")
    print("after epoch %d, test acc=%f" % (epoch, np.mean(acc_list)))
