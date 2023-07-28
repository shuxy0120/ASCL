import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import utils1


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores

def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss


def train(model, train_loader, eval_loader, args, device=torch.device("cuda"), qid2type=None):
    N = len(train_loader.dataset)
    resume = False
    lr_default = args.base_lr
    num_epochs = args.epochs
    optim = torch.optim.Adamax(model.parameters(), lr=lr_default)

    logger = utils1.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_score = 0

    utils1.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    last_eval_score, eval_score = 0, 0

    best_eval_score = 0
    for epoch in range(0, num_epochs):
        pbar = tqdm(total=len(train_loader))
        total_norm, count_norm = 0, 0
        total_loss, train_score = 0, 0
        count, average_loss, att_entropy = 0, 0, 0
        acc_all = AverageMeter()
        loss_pos = AverageMeter()
        loss_neg = AverageMeter()
        loss_pn = AverageMeter()
        t = time.time()
        logger.write('lr: %.6f' % optim.param_groups[-1]['lr'])
        last_eval_score = eval_score

        mini_batch_count = 0
        batch_multiplier = args.grad_accu_steps
        for i, (v, q, target, bias, bb, qids, label_index, hintscore, q_num) in enumerate(train_loader):
            if mini_batch_count == 0:
                optim.step()
                optim.zero_grad()
                mini_batch_count = batch_multiplier
            v = Variable(v).to(device)
            target = Variable(target).to(device)
            bias = Variable(bias).to(device)
            bb = Variable(bb).to(device)
            label_index = Variable(label_index).to(device)

            pred, loss1 = model(v, bb, q, target, bias, label_index, qid=q_num)
            loss = loss1[0]

            loss_pos.update(loss1[1].item(), v.size(0))
            loss_neg.update(loss1[2].item(), v.size(0))
            loss_pn.update(loss1[3].item(), v.size(0))
            
            loss.backward()

            mini_batch_count -= 1
            total_norm += nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
            count_norm += 1
            batch_score = compute_score_with_logits(pred, target, device).sum()
            total_loss += loss.data.item() * batch_multiplier * v.size(0)
            train_score += batch_score
            
            acc_all.update(batch_score, v.size(0))

            print('Train: [{0}/{1}]\t'
                  'Loss_pos {Loss_pos.val:.4f} ({Loss_pos.avg:.4f})\t'
                  'Loss_neg {Loss_neg.val:.4f} ({Loss_neg.avg:.4f})\t'
                  'Loss_pn {Loss_pn.val:.4f} ({Loss_pn.avg:.4f})\t'
                  'acc_all {acc_all.val:.3f} ({acc_all.avg:.3f})'.format(
                epoch, 20, Loss_pos=loss_pos,
                Loss_neg=loss_neg, Loss_pn=loss_pn, acc_all=acc_all))

            pbar.update(1)

            if args.log_interval > 0:
                average_loss += loss.data.item() * batch_multiplier
                count += 1
                if i % args.log_interval == 0:
                    att_entropy /= count
                    average_loss /= count
                    print("step {} / {} (epoch {}), ave_loss {:.3f},".format(
                            i, len(train_loader), epoch,
                            average_loss),
                          "att_entropy {:.3f}".format(att_entropy))
                    average_loss = 0
                    count = 0
                    att_entropy = 0

        print("train score:", 100 * train_score / N)
        total_loss /= N
        train_score = 100 * train_score / N
        if eval_loader is not None:
            eval_score, bound, entropy, results = evaluate(
                model, eval_loader, device, args, qid2type)
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f'
                     % (total_loss, total_norm / count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)'
                         % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

            if entropy is not None:
                info = ''
                for i in range(entropy.size(0)):
                    info = info + ' %.2f' % entropy[i]
                logger.write('\tentropy: ' + info)
        if (eval_loader is None)\
           or (eval_loader is not None and eval_score*100 > 68 and best_eval_score<eval_score): #  and eval_score > 68
            logger.write("saving current model weights to folder")
            model_path = os.path.join(args.output, 'model.pth')
            # model_path = os.path.join(args.output, 'model–%s.pth'%str(epoch))
            opt = optim if args.save_optim else None
            utils1.save_model(model_path, model, epoch, opt)
            best_eval_score = eval_score


@torch.no_grad()
def evaluate(model, dataloader, device, args, qid2type=None):
    model.eval()
    # relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    pbar = tqdm(total=len(dataloader))

    for i, (v, q, target, bias, bb, qids, label_index, hintscore, q_num) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        target = Variable(target).to(device)
        bias = Variable(bias).to(device)
        bb = Variable(bb).to(device)
        pred, _ = model(v, bb, q, None, None, mode='test')
        batch_score = compute_score_with_logits(
            pred, target, device)
        score += batch_score.sum()
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)

        for j in range(len(qids)):
            qid = qids[j].numpy()
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j].sum()
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j].sum()
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j].sum()
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

        pbar.update(1)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    results = dict(
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy, results


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        