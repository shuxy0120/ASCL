import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import json
from model.vqa_model import VQAModel

from dataset import Dictionary, VQAFeatureDataset
import model.regat as model
from train import compute_score_with_logits
from model.position_emb import prepare_graph_variables
from config.parser import Struct
import utils1

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def compute_score_with_logits2(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cpu()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

@torch.no_grad()
def evaluate(model, dataloader, model_hps, args, device):
    with open('util/qid2type_cpv2.json','r') as f:
        qid2type=json.load(f)
    model.eval()
    label2ans = dataloader.dataset.label2ans
    num_answers = len(label2ans)
    # relation_type = dataloader.dataset.relation_type
    N = len(dataloader.dataset)
    results = []
    results2 = []
    score = 0
    pbar = tqdm(total=len(dataloader))

    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    if args.save_logits:
        idx = 0
        pred_logits = np.zeros((N, num_answers))
        gt_logits = np.zeros((N, num_answers))

    for i, (v, q, target, bb, qid) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        # norm_bb = Variable(norm_bb).to(device)
        # q = Variable(q).to(device)
        target = Variable(target).to(device)
        bb = Variable(bb).to(device)
        pred, _ = model(v, bb, q, None, None)
        # Check if target is a placeholder or actual targets
        if target.size(-1) == num_answers:
            target = Variable(target).to(device)
            batch_score = compute_score_with_logits(
                pred, target, device)
            score += batch_score.sum()
            if args.save_logits:
                gt_logits[idx:batch_size+idx, :] = target.cpu().numpy()

            for j in range(len(qid)):
                qida = qid[j].numpy()
                typ = qid2type[str(qida)]
                if typ == 'yes/no':
                    score_yesno += batch_score[j].sum()
                    total_yesno += 1
                    if score_yesno == 0:
                        results2.append(qida)
                elif typ == 'other':
                    score_other += batch_score[j].sum()
                    total_other += 1
                    if score_other == 0:
                        results2.append(qida)
                elif typ == 'number':
                    score_number += batch_score[j].sum()
                    total_number += 1
                    if score_number == 0:
                        results2.append(qida)
                else:
                    print('Hahahahahahahahahahaha')

        if args.save_logits:
            pred_logits[idx:batch_size+idx, :] = pred.cpu().numpy()
            idx += batch_size

        if args.save_answers:
            qid = qid.cpu()
            pred = pred.cpu()
            current_results = make_json(pred, qid, dataloader)
            results.extend(current_results)

        pbar.update(1)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    score = score / N
    print("score_yesno:", score_yesno)
    print("score_number:", score_number)
    print("score_other:", score_other)
    print("score:", score)
    results_folder = f"{args.output_folder}/results"
    if args.save_logits:
        utils1.create_dir(results_folder)
        save_to = f"{results_folder}/logits_{args.dataset}" +\
            f"_{args.split}.npy"
        np.save(save_to, pred_logits)

        utils1.create_dir("./gt_logits")
        save_to = f"./gt_logits/{args.dataset}_{args.split}_gt.npy"
        if not os.path.exists(save_to):
            np.save(save_to, gt_logits)
    if args.save_answers:
        utils1.create_dir(results_folder)
        save_to = f"{results_folder}/{args.dataset}_" +\
            f"{args.split}.json"
        json.dump(results, open(save_to, "w"))
    filename = open('lmh-hw-5.txt', 'w')  
    for value in a:  
         filename.write(str(value)) 
    filename.close() 
    return score


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def make_json(logits, qIds, dataloader):
    utils1.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser()

    '''
    For eval logistics
    '''
    parser.add_argument('--save_logits', action='store_true', default=False,
                        help='save logits')
    parser.add_argument('--save_answers', action='store_true', default=False,
                        help='save poredicted answers')

    '''
    For loading expert pre-trained weights
    '''
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--output_folder', type=str, default="",
                        help="checkpoint folder")

    '''
    For dataset
    '''
    parser.add_argument('--data_folder', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='cpv2',
                        choices=["v2", "cpv1", "cpv2"])
    parser.add_argument('--split', type=str, default="test",
                        choices=["train", "val", "test", "test2015"],
                        help="test for vqa_cp, test2015 for vqa")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")

    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for eval" % (n_device))
    device = torch.device("cuda")

    dataset = args.dataset
    if dataset == 'cpv1':
        dictionary = Dictionary.load_from_file('../data/dictionary_v1.pkl')
    elif dataset == 'cpv2' or dataset == 'v2':
        dictionary = Dictionary.load_from_file('../data/dictionary.pkl')

    batch_size = 128

    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=False)

    model = VQAModel(2274).to(device)
    # model = nn.DataParallel(model).to(device)

    if args.checkpoint > 0:
        checkpoint_path = os.path.join(
                            args.output_folder,
                            f"model_{args.checkpoint}.pth")
    else:
        checkpoint_path = os.path.join(args.output_folder,
                                       f"model.pth")
    print("Loading weights from %s" % (checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise ValueError("No such checkpoint exists!")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get('model_state', checkpoint)
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    print("\tUnexpected_keys:", list(unexpected_keys))
    print("\tMissing_keys:", list(missing_keys))
    model.load_state_dict(matched_state_dict, strict=False)

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)

    eval_score = evaluate(
        model, eval_loader, None, args, device)

    print('\teval score: %.2f' % (100 * eval_score))
