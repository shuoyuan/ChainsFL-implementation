#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from json import dumps
import datetime
import pickle
import os
import pandas as pd
import datetime
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import buildModels

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dateNow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # # load dataset and split users
    # if args.dataset == 'mnist':
    #     trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    #     dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    #     # sample users
    #     if args.iid:
    #         # allocate the dataset index to users
    #         dict_users = mnist_iid(dataset_train, args.num_users)
    #     else:
    #         dict_users = mnist_noniid(dataset_train, args.num_users)
    # elif args.dataset == 'cifar':
    #     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #     if args.iid:
    #         dict_users = cifar_iid(dataset_train, args.num_users)
    #     else:
    #         exit('Error: only consider IID setting in CIFAR10')
    # else:
    #     exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # # build model
    # if args.model == 'cnn' and args.dataset == 'cifar':
    #     net_glob = CNNCifar(args=args).to(args.device)
    # elif args.model == 'cnn' and args.dataset == 'mnist':
    #     net_glob = CNNMnist(args=args).to(args.device)
    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')
    # print(net_glob)
    # net_glob.train()

    # build network
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    base_glob = torch.load('./data/genesisForCNN.pkl')
    net_glob.load_state_dict(base_glob)

    allDeviceName = []
    for i in range(args.num_users):
        allDeviceName.append("device"+("{:0>5d}".format(i)))

    # training
    acc_test_list = []
    loss_test_list = []

    acc_train_list = []
    loss_train_list = []

    # with open('D:\\expRes\\dict_users.pkl', 'rb') as f:
    #     dict_users = pickle.load(f)
    # idxs_users = [5, 56, 76, 78, 68, 25, 47, 15, 61, 55, 60, 37, 27, 70, 79, 34, 18, 88, 57, 98, 48, 46, 33, 82, 4, 7, 6, 91, 92, 52]
    # print('Number of selected devices '+str(len(idxs_users)))
    # idxs_users = [ 7, 85, 14, 67, 88, 72, 20, 77, 89, 34, 82, 15, 26, 6, 42, 8, 60 ,49, 65, 46, 53, 24 ,31 ,98 ,64, 13, 56, 19, 74, 95]
    m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        maliciousN = random.sample(idxs_users.tolist(), 0)
        # m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            if idx not in maliciousN:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                print('Training of '+str(allDeviceName[idx])+' in iteration '+str(iter)+' has done!')
            else:
                malN_glob = torch.load('./data/genesisForCNN.pkl', map_location=torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'))
                w_locals.append(copy.deepcopy(malN_glob))
                print('Training of malicious node '+str(allDeviceName[idx])+' in iteration '+str(iter)+' has done!')
        # update global weights
        w_glob = FedAvg(w_locals)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_list.append(loss_avg)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.eval()


        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_train, _ = test_img(net_glob, dataset_train, args)


        loss_test_list.append(loss_test)
        acc_test_list.append((acc_test.cpu().numpy().tolist()/100))

        accDfTest = pd.DataFrame({'baseline':acc_test_list})
        accDfTest.to_csv("D:\\ChainsFLexps\\ggFL\\GoogleFL-iid{}-{}-{}localEpochs-{}users-{}Rounds_ACC_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        lossDfTest = pd.DataFrame({'baseline':loss_test_list})
        lossDfTest.to_csv("D:\\ChainsFLexps\\ggFL\\GoogleFL-iid{}-{}-{}localEpochs-{}users-{}Rounds_Loss_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        # loss_train_list.append(loss_train)
        # acc_train_list.append((acc_train.cpu().numpy().tolist())/100)

        # accDfTrain = pd.DataFrame({'baseline':acc_train_list})
        # accDfTrain.to_csv("D:\\ChainsFLexps\\ggFL\\GoogleFL-Train-iid{}-{}-{}localEpochs-{}users-{}Rounds_ACC_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        # lossDfTrain = pd.DataFrame({'baseline':loss_train_list})
        # lossDfTrain.to_csv("D:\\ChainsFLexps\\ggFL\\GoogleFL-Train-iid{}-{}-{}localEpochs-{}users-{}Rounds_Loss_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

        # print('The acc in epoch '+str(iter)+' is '+str(acc_test.cpu().numpy().tolist()))
        # print('The loss in epoch '+str(iter)+' is '+str(loss_test))
        # # print('The content of w_glob', w_glob)
    
    print(acc_test_list)
    print(loss_test_list)

