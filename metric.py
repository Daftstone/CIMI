import numpy as np
import torch
import scipy
from scipy.stats.stats import kendalltau

from parse import *
from utils import *

args = parse_args(None)


def IOU_list(tokenizer, evidences, sentences, pred_mask):
    acc_list = []
    for i, (sen, ev) in enumerate(zip(sentences, evidences)):
        sen_ids = tokenizer(sen, max_length=args.max_length, truncation=True)['input_ids'][:args.max_length]
        ev_ids = tokenizer(ev, max_length=args.max_length, truncation=True)['input_ids'][1:-1]
        ev_ids = set(list(ev_ids)) & set(list(sen_ids))
        if (len(ev_ids) == 0):
            acc_list.append(0)
            return acc_list
        ids = sen_ids * (pred_mask[i, :len(sen_ids)] > 0.5)
        ids = ids[1:-1]
        sen_ids = ids[np.where(ids != 0)[0]]
        sen_ids = set(list(sen_ids))
        ev_ids = set(list(ev_ids))
        acc_list.append(len(sen_ids & ev_ids) / len(sen_ids | ev_ids))
    return acc_list


def difference_list(pred1, pred2, label):
    label = np.argmax(pred1, axis=1)
    diff_list = []
    for i in range(len(label)):
        diff_list.append(pred1[i, label[i]] - pred2[i, label[i]])
    return diff_list


def cal_corr(model, scores, batch_x, batch_y, lengths, device):
    labels = np.argmax(batch_y, axis=1)
    dims = len(scores[0])
    pred_import = np.zeros_like(scores)
    for i in range(dims):
        mask = np.ones_like(scores)
        mask[:, i] = 0.
        mask = torch.tensor(mask).float().to(device)
        preds = model.forward_mask(batch_x, mask).detach().cpu().numpy()

        for j in range(len(labels)):
            pred_import[j, i] = preds[j, labels[j]]
    cor_list = []
    for i in range(len(scores)):
        idx = np.argsort(scores[i, 1:lengths[i] - 1]) + 1
        score1 = scores[i, idx]
        score2 = pred_import[i, idx]
        if (len(score1) >= 2):
            cor_list.append(-scipy.stats.pearsonr(score1, score2)[0])
    return cor_list


def cal_sensitivity(model, scores, batch_x, lengths):
    same_list = []
    origin_list = []
    for j in range(len(batch_x)):
        idx = np.argsort(-scores[j, 1:lengths[j] - 1])[:10] + 1
        origin_list.append(idx)
    for i in range(10):
        mask, length = model.forward_noise(batch_x, 5)
        mask = mask.squeeze().detach().cpu().numpy()
        for j in range(len(batch_x)):
            idx = np.argsort(-mask[j, 1:length[j] - 1])[:10] + 1
            origin_idx = origin_list[j]
            same_list.append(len(set(list(idx)) & set(list(origin_idx))))
    return same_list


def cal_mono(model, scores, batch_x, batch_y, length, device):
    for i in range(len(scores)):
        scores[i, 0] = -1e9
        scores[i, length[i] - 1:] = -1e9
    labels = np.argmax(batch_y, axis=1)
    dims = len(scores[0])
    idx_sort = np.argsort(-scores, axis=-1)
    pred_import = np.zeros_like(scores)
    for i in range(dims):
        idx = idx_sort[:, :i]
        mask = np.ones_like(scores)
        for j in range(len(idx)):
            mask[j, idx[j]] = 0
        mask = torch.tensor(mask).float().to(device)
        preds = model.forward_mask(batch_x, mask).detach().cpu().numpy()

        for j in range(len(labels)):
            pred_import[j, i] = preds[j, labels[j]]
    cor_list = []
    for i in range(len(scores)):
        try:
            if (pred_import[j, 0] >= 0.5):
                idx = np.min(np.where(pred_import[j] < 0.5)[0])
                cor_list.append(np.minimum(idx, length[i] - 2))
            else:
                idx = np.min(np.where(pred_import[j] >= 0.5)[0])
                cor_list.append(np.minimum(idx, length[i] - 2))
        except:
            cor_list.append(length[i] - 2)
    return cor_list


def cal_mono(model, scores, batch_x, batch_y, length, device):
    for i in range(len(scores)):
        scores[i, 0] = -1e9
        scores[i, length[i] - 1:] = -1e9
    labels = np.argmax(batch_y, axis=1)
    dims = len(scores[0])
    idx_sort = np.argsort(-scores, axis=-1)
    pred_import = np.zeros_like(scores)
    for i in range(dims):
        idx = idx_sort[:, :i]
        mask = np.ones_like(scores)
        for j in range(len(idx)):
            mask[j, idx[j]] = 0
        mask = torch.tensor(mask).float().to(device)
        preds = model.forward_mask(batch_x, mask).detach().cpu().numpy()

        for j in range(len(labels)):
            pred_import[j, i] = preds[j, labels[j]]
    cor_list = []
    for i in range(len(scores)):
        try:
            if (pred_import[i, 0] >= 0.5):
                idx = np.min(np.where(pred_import[i] < 0.5)[0])
                cor_list.append(np.minimum(idx, length[i] - 2))
            else:
                idx = np.min(np.where(pred_import[i] >= 0.5)[0])
                cor_list.append(np.minimum(idx, length[i] - 2))
        except:
            cor_list.append(length[i] - 2)
    return cor_list


def cal_mono1(model, scores, batch_x, batch_y, length, device):
    for i in range(len(scores)):
        scores[i, 0] = -1e9
        scores[i, length[i] - 1:] = -1e9
    labels = np.argmax(batch_y, axis=1)
    dims = len(scores[0])
    idx_sort = np.argsort(-scores, axis=-1)
    pred_import = np.zeros_like(scores)
    for i in range(dims):
        idx = idx_sort[:, :i]
        mask = np.ones_like(scores)
        for j in range(len(idx)):
            mask[j, idx[j]] = 0
        mask = torch.tensor(mask).float().to(device)
        preds = model.forward_mask(batch_x, mask).detach().cpu().numpy()

        for j in range(len(labels)):
            pred_import[j, i] = preds[j, labels[j]]
    cor_list = []
    for i in range(len(scores)):
        try:
            if (pred_import[i, 0] >= 0.5):
                idx = np.min(np.where(pred_import[i] < 0.5)[0])
                if (idx > length[i] - 2):
                    cor_list.append(999999999)
                else:
                    cor_list.append(idx)
            else:
                idx = np.min(np.where(pred_import[i] >= 0.5)[0])
                if (idx > length[i] - 2):
                    cor_list.append(999999999)
                else:
                    cor_list.append(idx)
        except:
            cor_list.append(999999999)
    return cor_list
