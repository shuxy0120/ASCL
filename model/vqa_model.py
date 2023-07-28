# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.entry import LXRTEncoder
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import model.vqa_debias_loss_functions as vqa_loss_fc
import model.losses as lossfunc

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 16


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss

def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=3, dim=-1, sorted=False)
    # b = F.softmax(logits_neg, dim=-1)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss

def compute_self_loss2(logits_neg, a):
    pred_ind = torch.argsort(logits_neg, 1, descending=True)[:, :1]
    false_ans = torch.ones(logits_neg.shape[0], logits_neg.shape[1]).cuda()
    false_ans.scatter_(1, pred_ind.long(), 0)
    labels_neg = a * false_ans
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels_neg, dim=-1), k=1, dim=-1, sorted=False)
    # b = F.softmax(logits_neg, dim=-1)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind) * mask
    qice_loss = neg_top_k.sum(1).mean()
    return qice_loss


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.debias_loss_fn = vqa_loss_fc.LearnedMixin(0.36)

        # VQA Answer heads
        self.logit_fc = SimpleClassifier(in_dim=hid_dim, hid_dim=2 * hid_dim, out_dim=2274,
                                           dropout=0.5)


    def forward(self, feat, pos, sent, labels, bias, label_index=None, qid=None, mask=None, mode='train'):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x, pn_loss, x_neg = self.lxrt_encoder(sent, (feat, pos), visual_attention_mask=mask, qid=qid, mode=mode)
        logit = self.logit_fc(x)
        if mode=='train' and labels is not None and x_neg is not None:
            pred_ind = torch.argsort(logit, 1, descending=True)[:, :5]
            false_ans = torch.ones(logit.shape[0], logit.shape[1]).cuda()
            false_ans.scatter_(1, pred_ind.long(), 0)
            labels_neg = labels * false_ans

            logit_neg = self.logit_fc(x_neg)
            loss_neg = self.debias_loss_fn(x_neg, logit_neg, bias, labels_neg)

        if labels is not None:
            loss = self.debias_loss_fn(x, logit, bias, labels)
            if pn_loss is None:
                loss_all = loss
            else:
                loss_all = pn_loss + loss
            if mode=='train' and x_neg is not None:
                loss_all += loss_neg
        else:
            loss_neg = None
            pn_loss = None
            loss_all = None
            loss = None

        return logit, (loss_all, loss, loss_neg, pn_loss)


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
                                      dim=None))
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
                                  dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits