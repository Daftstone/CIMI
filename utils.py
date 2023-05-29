import torch
import pandas as pd
import numpy as np
import pickle
import collections
import torch.nn.functional as F
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from parse import *

args = parse_args(None)


def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        # os.environ['PYTHONHASHSEED'] = str(seed)


def generate_imdb_data():
    np.random.seed(100)
    data = pd.read_csv('data/IMDB/IMDB.csv')
    data = data.sample(len(data), replace=False)
    text = list(data['text'])
<<<<<<< HEAD
    evidences = list(data['text'])
=======
>>>>>>> a507ac7 (init)
    labels = np.array(list(data['label']))
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 2))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
<<<<<<< HEAD
    return text[:5000], evidences[:5000], labels_onehot[:5000]
=======
    return text[:5000], labels_onehot[:5000]
>>>>>>> a507ac7 (init)


def generate_hate_data():
    np.random.seed(100)
    data = pd.read_csv('data/hate/labeled_data.csv')
    data = data.sample(len(data), replace=False)
    text = list(data['tweet'])
<<<<<<< HEAD
    evidences = list(data['tweet'])
=======
>>>>>>> a507ac7 (init)
    labels = np.array(list(data['class']))
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 3))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
    idx3 = np.where(labels == 2)[0]
    labels_onehot[idx3, 2] = 1
<<<<<<< HEAD
    return text[:5000], evidences[:5000], labels_onehot[:5000]
=======
    return text[:5000], labels_onehot[:5000]
>>>>>>> a507ac7 (init)


def generate_yelp_data():
    np.random.seed(100)
    train_data = pd.read_csv("data/yelp/train.csv", encoding='utf-8-sig')
    test_data = pd.read_csv("data/yelp/test.csv", encoding='utf-8-sig')
    data = train_data.append(test_data)
    data = data.sample(len(data), replace=False)
    text = list(data['text'])
<<<<<<< HEAD
    evidences = list(data['text'])
=======
>>>>>>> a507ac7 (init)
    labels = np.array(list(data['label'])) - 1
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 2))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
<<<<<<< HEAD
    return text[:5000], evidences[:5000], labels_onehot[:5000]
=======
    return text[:5000], labels_onehot[:5000]
>>>>>>> a507ac7 (init)


def generate_clickbait_data():
    np.random.seed(100)
    data = pd.read_csv("data/clickbait/clickbait_data.csv", encoding='utf-8-sig')
    data = data.sample(len(data), replace=False)
    text = list(data['headline'])
<<<<<<< HEAD
    evidences = list(data['headline'])
=======
>>>>>>> a507ac7 (init)
    labels = np.array(list(data['clickbait']))
    idx = np.where(np.isnan(labels))[0]
    labels[idx] = 0
    labels_onehot = np.zeros((len(labels), 2))
    idx1 = np.where(labels == 0)[0]
    labels_onehot[idx1, 0] = 1
    idx2 = np.where(labels == 1)[0]
    labels_onehot[idx2, 1] = 1
<<<<<<< HEAD
    return text[:5000], evidences[:5000], labels_onehot[:5000]
=======
    return text[:5000], labels_onehot[:5000]
>>>>>>> a507ac7 (init)


def sequence_mask(lengths, max_len, device):
    x = torch.arange(max_len).expand(lengths.size()[0], max_len).to(device)
    y = lengths.unsqueeze_(-1).expand(-1, max_len) - 1

    mask = torch.le(input=x.long(), other=y.long())

    return mask


def generator_loss(mask, length, device, args):
    mask = mask.squeeze()
    mask_for_valid = sequence_mask(length, mask.size()[1], device).float()

    valid_mask = mask * mask_for_valid

    selection_loss = torch.sum(valid_mask, dim=1) / length.squeeze_().float()
    # selection_loss = torch.maximum(torch.ones_like(selection_loss).to(device) * 0.2, selection_loss)

    padding = torch.zeros(mask.size()[0], 1).to(device)

    mask_shift_right = torch.cat([padding, mask[:, :-1]], dim=1)
    transitions = torch.abs(mask - mask_shift_right).float()
    transitions *= mask_for_valid

    transitions_loss = torch.sum(transitions, dim=1) / length.float()

    generator_loss = args.theta * selection_loss + args.gamma * transitions_loss

    return selection_loss, transitions_loss, generator_loss, valid_mask, mask_for_valid


def get_evidence_ids(tokenizer, evidence, sentence, length):
    ids_list = []
    sen_ids = tokenizer(sentence)['input_ids'][:args.max_length]
    ev_ids = tokenizer(evidence)['input_ids'][1:length - 1]
    for ev in ev_ids:
        idx = np.where(np.array(sen_ids) == ev)[0]
        if (len(idx) > 0):
            ids_list += list(idx)
    return ids_list


def word2ids(sentence, word_map, tokenizer, length=512):
    sen_ids = np.array(tokenizer(sentence)['input_ids'][:length])
    id_map = collections.defaultdict(lambda: 0)
    for key, value in word_map.items():
        tokens = tokenizer(key)['input_ids'][1:-1]
        for token in tokens:
            id_map[token] = value
    ids_list = np.zeros(length)
    for i, ids in enumerate(sen_ids):
        ids_list[i] = id_map[ids]
    return ids_list


def get_gradient_mask(embedding, loss):
    gradients = torch.autograd.grad(loss, embedding)
    gradients = torch.sum(torch.abs(gradients[0]), dim=-1)
    gradients = gradients.detach().cpu().numpy()
    return gradients


def get_stack_mask(valid_mask):
    return valid_mask.detach().cpu().numpy()


def get_explain(batch_x, batch_y, model, scores, cur_explain, lengths, ratio):
    encoded_input = model.tokenizer(batch_x, padding=True, max_length=args.max_length, truncation=True)
    label = np.argmax(batch_y, axis=-1)
    for i in range(len(batch_x)):
        idx = np.argsort(-scores[i, 1:lengths[i] - 1])[:int(lengths[i] * ratio)] + 1
        token_ids = np.array(encoded_input['input_ids'][i])
        tokens = model.tokenizer.convert_ids_to_tokens(list(token_ids))
        for ii in idx:
            tokens[ii] = "<" + tokens[ii] + ">"
        cur_explain = cur_explain.append({'text': batch_x[i], 'explain': " ".join(tokens), 'label': label[i]},
                                         ignore_index=True)
    return cur_explain


def find_ratio(scores, lengths, target, left, right):
    ratio = (left + right) / 2
    len_list = []
    for scs, lens in zip(scores, lengths):
        for score, length in zip(scs, lens):
            idx = np.where(score[1:length - 1] >= ratio)[0] + 1
            len_list.append(len(idx))
    # print(np.mean(len_list), ratio, target)
    if (np.mean(len_list) - target < -1):
        return find_ratio(scores, lengths, target, left, ratio)
    elif (np.mean(len_list) - target > 1):
        return find_ratio(scores, lengths, target, ratio, right)
    else:
        return ratio


def generate_sim(embed1, embed2):
    embeddings1 = embed1
    embeddings2 = embed2
    pos_sim = torch.cosine_similarity(embeddings1, embeddings2, dim=0)
    return -pos_sim.mean()


def generate_dis(embed1, embed2):
    embeddings1 = embed1
    embeddings2 = embed2
    pos_sim = torch.mean(torch.square(embeddings1 - embeddings2))
    return pos_sim * len(embeddings1)


def generate_dis1(embed1, embed2):
    embeddings1 = embed1
    embeddings2 = embed2
    pos_sim = torch.cosine_similarity(embeddings1, embeddings2, dim=-1).sum()
    return -pos_sim


def generate_mse_loss(pred1, pred2, batch_y):
    labels = torch.argmax(batch_y, dim=-1).cpu().numpy()
    dis = 0.
    for i in range(batch_y.size()[1]):
        idx = np.where(labels == i)[0]
        dis += torch.square(pred1[idx, i] - batch_y[idx, i]).sum() - torch.square(pred2[idx, i] - batch_y[idx, i]).sum()
    return dis


def map_weights(weight, origin, target):
    new_weights = {}
    for key, value in weight.items():
        if (key[:len(origin)] == origin):
            new_weights[target + key[len(origin)]] = value
        else:
            new_weights[key] = value
    return new_weights