import argparse
from model import Code2Vec
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import C2VDataSet
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import tqdm
import math
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of examples in epoch')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--terminal_embed_size', type=int, default=128,
                        help='embedding size of terminal')
    # bilstmなので*2
    parser.add_argument('--path_embed_size', type=int, default=128 * 2,
                        help='embedding size of path')
    parser.add_argument('--datapath',
                        default="./java-small/java-small.dict.c2v",
                        help='path of input data')
    parser.add_argument('--trainpath',
                        default="./data/java14m/java14m.train.c2v",
                        help='path of train data')
    parser.add_argument('--validpath',
                        default="./data/java14m/java14m.val.c2v",
                        help='path of valid data')
    parser.add_argument('--savename', default="",
                        help='name of saved model')
    parser.add_argument('--trainnum', type=int, default=15344512,
                        help='size of train data')
    parser.add_argument('--validnum', type=int, default=320866,
                        help='size of valid data')
    parser.add_argument('--context_length', type=int, default=200,
                        help='length of context')
    parser.add_argument('--terminal_length', type=int, default=5,
                        help='length of terminal')
    parser.add_argument('--path_length', type=int, default=9,
                        help='length of path')
    parser.add_argument('--target_length', type=int, default=7,
                        help='length of target')
    parser.add_argument('--eval', action="store_true",
                        help='is eval')
    parser.add_argument('--path_rnn_drop', type=float, default=0.5,
                        help='drop rate of path rnn')
    parser.add_argument('--embed_drop', type=float, default=0.25,
                        help='drop rate of embbeding')
    parser.add_argument('--num_worker', type=int, default=0,
                        help='the number of worker')

    args = parser.parse_args()

    device = torch.device(
        args.gpu if args.gpu != -1 and torch.cuda.is_available() else "cpu")
    print(device)

    with open(args.datapath, 'rb') as file:
        terminal_counter = pickle.load(file)
        path_counter = pickle.load(file)
        target_counter = pickle.load(file)
        # _ = pickle.load(file)
        # _ = pickle.load(file)
        print('Dictionaries loaded.')
    train_h5 = h5py.File(args.trainpath, 'r')
    test_h5 = h5py.File(args.validpath, 'r')

    terminal_dict = {w: i for i, w in enumerate(
        sorted([w for w, c in terminal_counter.items() if c > 0]))}
    terminal_dict["<unk>"] = len(terminal_dict)
    terminal_dict["<pad>"] = len(terminal_dict)
    path_dict = {w: i for i, w in enumerate(sorted(path_counter.keys()))}
    path_dict["<unk>"] = len(path_dict)
    path_dict["<pad>"] = len(path_dict)
    target_dict = {w: i for i, w in enumerate(
        sorted([w for w, c in target_counter.items() if c > 0]))}
    target_dict["<unk>"] = len(target_dict)
    target_dict["<bos>"] = len(target_dict)
    target_dict["<pad>"] = len(target_dict)

    print("terminal_vocab:", len(terminal_dict))
    print("target_vocab:", len(target_dict))

    c2v = Code2Vec(args, terminal_vocab_size=len(terminal_dict),
                   path_element_vocab_size=len(path_dict),
                   target_dict=target_dict, device=device)\
        .to(device)

    if args.resume != "":
        c2v.load_state_dict(torch.load(args.resume))

    trainloader = DataLoader(
        C2VDataSet(args, train_h5, args.trainnum,
                   terminal_dict, path_dict, target_dict, device),
        batch_size=args.batchsize,
        shuffle=True, num_workers=args.num_worker)

    validloader = DataLoader(
        C2VDataSet(args, test_h5, args.validnum,
                   terminal_dict, path_dict, target_dict, device),
        batch_size=args.batchsize,
        shuffle=True, num_workers=args.num_worker)

    optimizer = optim.SGD(c2v.parameters(), lr=0.01,
                          momentum=0.95)
    # scheduler = optim.lr_scheduler.LambdaLR(
    #    optimizer,
    #    lr_lambda=lambda e: 0.01 * pow(0.95, (e * args.batchsize / args.trainnum)))

    for epoch in range(args.epoch):
        if not args.eval:
            sum_loss = 0
            train_count = 0
            c2v.train()
            for data in tqdm.tqdm(trainloader):
                # scheduler.step()
                optimizer.zero_grad()
                loss = c2v(*data, is_eval=False)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                if train_count % 250 == 0 and train_count != 0:
                    print(sum_loss / 250)
                    sum_loss = 0
                train_count += 1
            sum_loss = 0
        true_positive, false_positive, false_negative = 0, 0, 0
        for data in tqdm.tqdm(validloader):
            c2v.eval()
            with torch.no_grad():
                true_positive_, false_positive_, false_negative_ = c2v(
                    *data, is_eval=True)
            true_positive += true_positive_
            false_positive += false_positive_
            false_negative += false_negative_

        pre_score, rec_score, f1_score = calculate_results(
            true_positive, false_positive, false_negative)
        print("f1:", f1_score, "prec:", pre_score,
              "rec:", rec_score)
        if args.eval:
            break
        if args.savename != "":
            torch.save(c2v.state_dict(), args.savename + str(epoch) + ".model")
    if args.savename != "":
        torch.save(c2v.state_dict(), args.savename + ".model")


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


if __name__ == "__main__":
    # torch.manual_seed(23)
    # try:
    #    torch.multiprocessing.set_start_method("spawn")
    # except RuntimeError:
    #    pass
    main()
