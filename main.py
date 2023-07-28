import os
from os.path import join, exists
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import random
import json
from collections import defaultdict, Counter
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
from config.parser import parse_with_config
from train import train
import utils1
from utils1 import trim_collate
from model.vqa_model import VQAModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--grad_accu_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', type=str, default='../saved_models/')
    parser.add_argument('--save_optim', action='store_true',
                        help='save optimizer')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--dataset', type=str, default='cpv2',
                        choices=["v2", "cpv2", "cpv1"])
    parser.add_argument('--data_folder', type=str, default='../data')
    parser.add_argument('--load_lxmert', type=str, default='../data/pretrained/pretrained_LXRT.pth')
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true',
                        help='use visual genome dataset to train?')
    parser.add_argument('--adaptive', action='store_true',
                        help='adaptive or fixed number of regions')
    parser.add_argument('--name', type=str, default='AdvCl_cpv2')
    parser.add_argument('--cache_features', default=False)

    args = parse_with_config(parser)
    return args

def get_bias(train_dset,eval_dset):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset,eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]


if __name__ == '__main__':
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")

    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for training" % (n_device))
    device = torch.device("cuda")
    batch_size = args.batch_size

    torch.backends.cudnn.benchmark = True

    if args.seed != -1:
        print("Predefined randam seed %d" % args.seed)
    else:
        # fix seed
        args.seed = random.randint(1, 10000)
        print("Choose random seed %d" % args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = VQAModel(2274).to(device)
    if args.load_lxmert is not None:
        model.lxrt_encoder.load(args.load_lxmert)

    dataset = args.dataset
    if dataset == 'cpv1':
        dictionary = Dictionary.load_from_file('../data/dictionary_v1.pkl')
    elif dataset == 'cpv2' or dataset == 'v2':
        dictionary = Dictionary.load_from_file('../data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                   cache_image_features=args.cache_features)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)

    get_bias(train_dset, eval_dset)
    with open(join('util/qid2type_%s.json' % args.dataset), 'r') as f:
        qid2type = json.load(f)

    if args.checkpoint != "":
        print("Loading weights from %s" % (args.checkpoint))
        if not os.path.exists(args.checkpoint):
            raise ValueError("No such checkpoint exists!")
        checkpoint = torch.load(args.checkpoint)
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
        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        new_dict = {k: v for k, v in matched_state_dict.items() if k in model.state_dict().keys()}
        model.load_state_dict(matched_state_dict, strict=False)

    output_meta_folder = join(args.output, "ASCL_%s" % args.name)
    utils1.create_dir(output_meta_folder)
    args.output = output_meta_folder+"/%s_%s_%d" % (
                args.name, args.dataset, args.seed)
    if exists(args.output) and os.listdir(args.output):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output))
    utils1.create_dir(args.output)
    with open(join(args.output, 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    logger = utils1.Logger(join(args.output, 'log.txt'))

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)

    train(model, train_loader, eval_loader, args, device, qid2type)
