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

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

get_data_map = {'yelp': generate_yelp_data, 'clickbait': generate_clickbait_data, 'imdb': generate_imdb_data,
                'hate': generate_hate_data}
get_batch_map = {'yelp': 4, 'clickbait': 8, 'imdb': 4, 'hate': 8}

args = parse_args(None)
device = torch.device("cuda", args.device)
setup_seed()

text, evidence, label = get_data_map[args.dataset]()
split = int(len(text) * 0.8)
train_titles, train_evidences, train_labels = text[:split], evidence[:split], label[:split]
test_titles, test_evidences, test_labels = text[split:], evidence[split:], label[split:]

train_dataset = My_dataset(train_titles, train_evidences, train_labels)
test_dataset = My_dataset(test_titles, test_evidences, test_labels)

model = Bert_stack(args).to(device)
weights = torch.load(f"save/{args.dataset}_stack/{args.dataset}_stack.pt", map_location=device)
model.load_state_dict(weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_ce = torch.nn.BCELoss(reduction='none')

best_acc = 0.
best_mse = 100.
pre_epoch = 5

pred_list = []
label_list = []

cor_list = []
dffot_list = []
sen_list = []
all_comp = [[] for i in range(5)]
all_suf = [[] for i in range(5)]
all_IOU = [[] for i in range(5)]
all_length = [[] for i in range(5)]
all_mask = [[] for i in range(5)]
cur_explain = [pd.DataFrame(columns=['text', 'explain', 'label']) for i in range(101)]

model.eval()
loader = Data.DataLoader(dataset=test_dataset, batch_size=get_batch_map[args.dataset], shuffle=False, num_workers=2)
pbar = tqdm.tqdm(loader, "test", total=len(test_dataset) // get_batch_map[args.dataset])
for cur_batch, (batch_x, batch_ev, batch_y) in enumerate(pbar):
    pred1, mask, lengths, detail_output = model(batch_x)
    pred, _, _ = model.forward_single(batch_x)
    batch_y = batch_y.to(device)

    pred_true = torch.gt(pred, 0.5).float()
    loss = loss_ce(pred, pred_true).sum()

    batch_y = batch_y.cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred_list.append(pred)
    label_list.append(batch_y)

    selection_loss, transitions_loss, g_loss, valid_mask, valid_mask_flag = generator_loss(mask, lengths, device,
                                                                                               args)
    scores = get_stack_mask(valid_mask.squeeze())

    scores += np.random.rand(scores.shape[0], scores.shape[1]) * 1e-5
    lengths = lengths.detach().cpu().numpy()

    dffot_list += cal_mono(model, scores, batch_x, batch_y, lengths, device)
    sen_list += cal_sensitivity(model, scores, batch_x, lengths)

    for ii, topk in enumerate([1, 5, 10, 20, 50]):
        valid_mask_com = np.ones(detail_output[2][0][:, :, 0].size())
        valid_mask_suf = np.ones(detail_output[2][0][:, :, 0].size())
        count = []
        for i in range(len(batch_x)):
            length = lengths[i]
            idx = np.argsort(-scores[i, 1:length - 1])[:topk] + 1
            valid_mask_com[i, idx] = 0
            idx2 = np.argsort(-scores[i, 1:length - 1])[topk:] + 1
            valid_mask_suf[i, idx2] = 0
            count.append(len(idx))
            all_mask[ii].append(np.argsort(-scores[i, 1:length - 1]))

        valid_mask_com = torch.tensor(valid_mask_com).float().to(device)
        valid_mask_suf = torch.tensor(valid_mask_suf).float().to(device)
        pred_comp = model.forward_mask(batch_x, valid_mask_com).detach().cpu().numpy()
        pred_suff = model.forward_mask(batch_x, valid_mask_suf).detach().cpu().numpy()

        all_comp[ii] += difference_list(pred, pred_comp, batch_y)
        all_suf[ii] += difference_list(pred, pred_suff, batch_y)
        all_length[ii] += count
        cur_explain[ii] = get_explain(batch_x, batch_y, model, scores, cur_explain[ii], lengths, ii / 20)
all_comp = [np.mean(i) for i in all_comp]
all_suf = [np.mean(i) for i in all_suf]
all_length = [np.mean(i) for i in all_length]

print("sensitivity:", np.mean(sen_list))
print("MONO:", np.mean(dffot_list))
print("comp", np.mean(all_comp))
print("suf", np.mean(all_suf))
