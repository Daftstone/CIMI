import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig
# from transformers import DistilBertModel as BertModel
# from transformers import DistilBertTokenizer as BertTokenizer
# from transformers import DistilBertConfig as BertConfig
import torch.nn.functional as F
import torch.autograd as autograd
import time
import pandas as pd


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = torch.device("cuda", args.device)
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.fc_e1 = nn.Linear(768, 64)
        if (self.args.dataset == "hate"):
            self.fc_e2 = nn.Linear(64, 3)
        else:
            self.fc_e2 = nn.Linear(64, 2)

    def forward(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.args.max_length,
                                       truncation=True).to(self.device)
        output = self.bert(**encoded_input)
        fc = output[1]
        fc = F.elu(self.fc_e1(fc))
        fc = F.softmax(self.fc_e2(fc), dim=-1)
        return fc, output, torch.sum(encoded_input['attention_mask'], dim=1)


class Bert_stack(nn.Module):
    def __init__(self, args):
        super(Bert_stack, self).__init__()
        self.args = args
        self.device = torch.device("cuda", args.device)
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = args.max_length

        self.fc_g1 = nn.LSTM(input_size=768 * 2, hidden_size=64, batch_first=True, num_layers=1)
        self.fc_g2 = nn.Linear(64, 16)
        self.fc_g3 = nn.Linear(16, 2)
        self.fc_e1 = nn.Linear(768, 64)
        if (self.args.dataset == "hate"):
            self.fc_e2 = nn.Linear(64, 3)
        else:
            self.fc_e2 = nn.Linear(64, 2)
        self.sigma1 = nn.Parameter(torch.Tensor([1]))
        self.sigma2 = nn.Parameter(torch.Tensor([1]))

        # self.train_nn = []
        # for param in self.fc_g1.parameters():
        #     self.train_nn.append(param)
        # for param in self.fc_g2.parameters():
        #     self.train_nn.append(param)

        if (args.train_stack):
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.fc_e1.parameters():
                param.requires_grad = False
            for param in self.fc_e2.parameters():
                param.requires_grad = False

    def sample(self, probs):

        if self.training:
            mask = probs
        else:
            mask = probs
        return mask

    def sequence_mask(self, lengths, max_len, device):
        x = torch.arange(max_len).expand(lengths.size()[0], max_len).to(device)
        y = lengths.unsqueeze_(-1).expand(-1, max_len) - 2

        mask = torch.le(input=x.long(), other=y.long())
        mask[:, 0] = 0.

        return mask

    def forward(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(self.device)
        lengths = torch.sum(encoded_input['attention_mask'], dim=1)
        output = self.bert(**encoded_input)
        # output_bottom = torch.cat([output[2][0], output[1].unsqueeze(1).repeat(1, output[0].size()[1], 1)], dim=-1)
        output_bottom = torch.cat([output[2][0], output[0]], dim=-1)
        hidden1 = self.fc_g1(output_bottom)[0]
        hidden1 = F.leaky_relu(self.fc_g2(hidden1))
        logits = self.fc_g3(hidden1)
        # probs_read = F.gumbel_softmax(logits, 1.0)[:, :, 1:2]
        probs_read = F.softmax(logits, dim=-1)[:, :, 1:2]
        mask = self.sample(probs_read).float()
        valid_mask = self.sequence_mask(lengths, mask.size()[1], self.device).unsqueeze(-1)

        embedding_output = output[2][0]
        embedding_output_mask = embedding_output * ((mask - 1.) * valid_mask + 1.)
        extended_attention_mask = self.bert.get_extended_attention_mask(encoded_input['attention_mask'],
                                                                        encoded_input['input_ids'].shape,
                                                                        self.device)
        encoder_outputs = self.bert.encoder(embedding_output_mask, attention_mask=extended_attention_mask,
                                            output_attentions=self.bert.config.output_attentions,
                                            output_hidden_states=self.bert.config.output_hidden_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        fc_drop = outputs[1]
        fc_elu = F.elu(self.fc_e1(fc_drop))
        fc = F.softmax(self.fc_e2(fc_elu), dim=-1)
        return fc, mask, torch.sum(encoded_input['attention_mask'], dim=1), outputs

    def forward_mask(self, x, input_mask):
        if (len(input_mask.size()) == 2):
            input_mask = input_mask.unsqueeze(2)
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(
            self.device)

        embedding_output = self.bert.embeddings(input_ids=encoded_input['input_ids'],
                                                token_type_ids=encoded_input['token_type_ids'])
        embedding_output_mask = embedding_output * input_mask
        extended_attention_mask = self.bert.get_extended_attention_mask(encoded_input['attention_mask'],
                                                                        encoded_input['input_ids'].shape,
                                                                        self.device)
        encoder_outputs = self.bert.encoder(embedding_output_mask, attention_mask=extended_attention_mask,
                                            output_attentions=self.bert.config.output_attentions,
                                            output_hidden_states=self.bert.config.output_hidden_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        fc_drop = outputs[1]
        fc_elu = F.elu(self.fc_e1(fc_drop))
        fc = F.softmax(self.fc_e2(fc_elu), dim=-1)
        return fc

    def forward_lime(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(
            self.device)
        outputs = self.bert(**encoded_input)
        fc_drop = outputs[1]
        fc_elu = F.elu(self.fc_e1(fc_drop))
        fc = F.softmax(self.fc_e2(fc_elu), dim=-1)
        return fc.detach().cpu().numpy()

    def forward_single(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(
            self.device)
        outputs = self.bert(**encoded_input)
        fc_drop = outputs[1]
        fc_elu = F.elu(self.fc_e1(fc_drop))
        fc = F.softmax(self.fc_e2(fc_elu), dim=-1)
        return fc, torch.sum(encoded_input['attention_mask'], dim=1), outputs

    def forward_dis(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(self.device)
        lengths = torch.sum(encoded_input['attention_mask'], dim=1)
        output = self.bert(**encoded_input)
        output_bottom = torch.cat([output[2][0], output[0]], dim=-1)
        hidden1 = self.fc_g1(output_bottom)[0]
        hidden1 = F.leaky_relu(self.fc_g2(hidden1))
        logits = self.fc_g3(hidden1)
        probs_read = F.softmax(logits, dim=-1)[:, :, 1:2]
        mask = self.sample(probs_read).float()
        valid_mask = self.sequence_mask(lengths, mask.size()[1], self.device).unsqueeze(-1)

        ii = np.random.randint(output[2][0].size()[0])
        neg_embed = torch.cat([output[2][0][ii:], output[2][0][:ii]], dim=0)
        output_bottom1 = torch.cat([neg_embed, output[0]], dim=-1)
        hidden_neg1 = self.fc_g1(output_bottom1)[0]
        hidden_neg1 = F.leaky_relu(self.fc_g2(hidden_neg1))
        logits1 = self.fc_g3(hidden_neg1)
        probs_read1 = F.softmax(logits1, dim=-1)[:, :, 1:2]
        neg_mask = self.sample(probs_read1).float()

        embedding_output = output[2][0]
        embedding_output_pos = embedding_output * ((mask - 1.) * valid_mask + 1.)
        extended_attention_mask = self.bert.get_extended_attention_mask(encoded_input['attention_mask'],
                                                                        encoded_input['input_ids'].shape,
                                                                        self.device)
        encoder_outputs = self.bert.encoder(embedding_output_pos, attention_mask=extended_attention_mask,
                                            output_attentions=self.bert.config.output_attentions,
                                            output_hidden_states=self.bert.config.output_hidden_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        fc_drop = outputs[1]
        fc_elu = F.elu(self.fc_e1(fc_drop))
        fc_pos = F.softmax(self.fc_e2(fc_elu), dim=-1)

        embedding_output1 = output[2][0]
        embedding_output_neg = embedding_output1 * ((1. - mask - 1.) * valid_mask + 1.)
        extended_attention_mask1 = self.bert.get_extended_attention_mask(encoded_input['attention_mask'],
                                                                         encoded_input['input_ids'].shape,
                                                                         self.device)
        encoder_outputs1 = self.bert.encoder(embedding_output_neg, attention_mask=extended_attention_mask1)
        sequence_output1 = encoder_outputs1[0]
        pooled_output1 = self.bert.pooler(sequence_output1)
        outputs1 = (sequence_output1, pooled_output1,) + encoder_outputs1[1:]
        fc_drop1 = outputs1[1]
        fc_elu1 = F.elu(self.fc_e1(fc_drop1))
        fc_neg = F.softmax(self.fc_e2(fc_elu1), dim=-1)

        output_bottom1 = torch.cat([outputs[2][0], outputs[0]], dim=-1)
        hidden11 = self.fc_g1(output_bottom1)[0]
        hidden11 = F.leaky_relu(self.fc_g2(hidden11))
        logits1 = self.fc_g3(hidden11)
        return fc_pos, fc_neg, mask, torch.sum(encoded_input['attention_mask'],
                                               dim=1), logits, logits1, mask, neg_mask

    def forward_dis1(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(self.device)
        lengths = torch.sum(encoded_input['attention_mask'], dim=1)
        output = self.bert(**encoded_input)
        output_bottom = torch.cat([output[2][0], output[0]], dim=-1)
        hidden1 = self.fc_g1(output_bottom)[0]
        hidden1 = F.leaky_relu(self.fc_g2(hidden1))
        logits = self.fc_g3(hidden1)
        probs_read = F.softmax(logits, dim=-1)[:, :, 1:2]
        mask = self.sample(probs_read).float()
        valid_mask = self.sequence_mask(lengths, mask.size()[1], self.device).unsqueeze(-1)

        ii = np.random.randint(output[2][0].size()[0])
        neg_embed = torch.cat([output[2][0][ii:], output[2][0][:ii]], dim=0)
        output_bottom1 = torch.cat([neg_embed, output[0]], dim=-1)
        hidden_neg1 = self.fc_g1(output_bottom1)[0]
        hidden_neg1 = F.leaky_relu(self.fc_g2(hidden_neg1))
        logits1 = self.fc_g3(hidden_neg1)
        probs_read1 = F.softmax(logits1, dim=-1)[:, :, 1:2]
        neg_mask = self.sample(probs_read1).float()

        embedding_output = output[2][0]
        embedding_output_pos = embedding_output * ((mask - 1.) * valid_mask + 1.)
        extended_attention_mask = self.bert.get_extended_attention_mask(encoded_input['attention_mask'],
                                                                        encoded_input['input_ids'].shape,
                                                                        self.device)
        encoder_outputs = self.bert.encoder(embedding_output_pos, attention_mask=extended_attention_mask,
                                            output_attentions=self.bert.config.output_attentions,
                                            output_hidden_states=self.bert.config.output_hidden_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        fc_drop = outputs[1]
        fc_elu = F.elu(self.fc_e1(fc_drop))
        fc_pos = F.softmax(self.fc_e2(fc_elu), dim=-1)

        embedding_output1 = output[2][0]
        embedding_output_neg = embedding_output1 * ((1. - mask - 1.) * valid_mask + 1.)
        extended_attention_mask1 = self.bert.get_extended_attention_mask(encoded_input['attention_mask'],
                                                                         encoded_input['input_ids'].shape,
                                                                         self.device)
        encoder_outputs1 = self.bert.encoder(embedding_output_neg, attention_mask=extended_attention_mask1)
        sequence_output1 = encoder_outputs1[0]
        pooled_output1 = self.bert.pooler(sequence_output1)
        outputs1 = (sequence_output1, pooled_output1,) + encoder_outputs1[1:]
        fc_drop1 = outputs1[1]
        fc_elu1 = F.elu(self.fc_e1(fc_drop1))
        fc_neg = F.softmax(self.fc_e2(fc_elu1), dim=-1)

        get_noise_map = {'religion': 0.2, 'rotten': 0.2, 'yelp': 0.2, 'clickbait': 0.1, 'sentiment': 0.1, 'food': 0.2,
                         'imdb': 0.2, 'hate': 0.1}
        vmask = (mask - 1.) * valid_mask + 1.
        embedding_output1 = output[2][0]
        ii = np.random.randint(embedding_output1.size()[0])
        embedding_output2 = torch.cat([embedding_output1[ii:], embedding_output1[:ii]], dim=0)
        lam = torch.tensor(np.random.random((embedding_output2.size()[0], 1, 1))).to(self.device).float() * \
              get_noise_map[self.args.dataset]
        embedding_noise = vmask * embedding_output1 + (1 - vmask) * (
                (1 - lam) * embedding_output1 + lam * embedding_output2)
        encoder_outputs = self.bert.encoder(embedding_noise, attention_mask=extended_attention_mask,
                                            output_attentions=self.bert.config.output_attentions,
                                            output_hidden_states=self.bert.config.output_hidden_states)
        output_bottom2 = torch.cat([embedding_noise, encoder_outputs[0]], dim=-1)
        hidden2 = self.fc_g1(output_bottom2)[0]
        hidden2 = F.leaky_relu(self.fc_g2(hidden2))
        logits2 = self.fc_g3(hidden2)
        return fc_pos, fc_neg, mask, torch.sum(encoded_input['attention_mask'], dim=1), logits, logits2, mask, neg_mask

    def forward_noise(self, x, length):
        encoded_input = self.tokenizer(x, return_tensors='pt', padding=True, max_length=self.max_length,
                                       truncation=True).to(
            self.device)
        ids = encoded_input['input_ids']
        ids1 = torch.cat([ids[1:], ids[:1]], dim=0)
        for i in range(len(ids)):
            idx = np.random.choice(np.arange(len(ids[0])), length)
            ids[i, idx] = ids1[i, idx]
        encoded_input['input_ids'] = ids
        lengths = torch.sum(encoded_input['attention_mask'], dim=1)
        output = self.bert(**encoded_input)
        output_bottom = torch.cat([output[2][0], output[0]], dim=-1)
        hidden1 = self.fc_g1(output_bottom)[0]
        hidden1 = F.leaky_relu(self.fc_g2(hidden1))
        logits = self.fc_g3(hidden1)
        probs_read = F.softmax(logits, dim=-1)[:, :, 1:2]
        mask = self.sample(probs_read).float()
        return mask, lengths