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
import pandas as pd
import os
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


dateNow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            # allocate the dataset index to users
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    tmp_glob = torch.load('./data/genesisGPUForCNN.pkl')


    # training
    loss_train = []

    m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # malicious node select randomly
    random.seed(10)
    maliciousN = random.sample(idxs_users.tolist(), 2)
    print("The malicious nodes are " + str(maliciousN))

    workerIterIdx = {}
    
    for item in idxs_users:
        workerIterIdx[item] = 0

    realEpoch = 0
    currentEpoch = 0

    acc_test_list = []
    loss_test_list = []

    acc_train_list = []
    loss_train_list = []

    while currentEpoch <= args.epochs:
        currentEpoch += 1
        w_fAvg = []
        base_glob = tmp_glob
        net_glob.load_state_dict(base_glob)

        print('# of current epoch is ' + str(currentEpoch))

        workerNow = np.random.choice(idxs_users, 1, replace=False).tolist()[0]

        staleFlag = np.random.randint(-1,4,size=1)

        print('The staleFlag of worker ' + str(workerNow) + ' is ' + str(staleFlag))

        if staleFlag <= 4:

            # judge the malicious node 
            if workerNow not in maliciousN:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[workerNow])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            else:
                w = torch.load('./data/genesisGPUForCNN.pkl', map_location=torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'))
                print('Training of malicious node device '+str(workerNow)+' in iteration '+str(currentEpoch)+' has done!')

            # means that the alpha is 0.5
            w_fAvg.append(copy.deepcopy(base_glob))
            w_fAvg.append(copy.deepcopy(w))
            tmp_glob = FedAvg(w_fAvg)

            net_glob.load_state_dict(tmp_glob)
            net_glob.eval()
            
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            # acc_train, loss_train = test_img(net_glob, dataset_train, args)

            acc_test_list.append(acc_test.cpu().numpy().tolist()/100)
            loss_test_list.append(loss_test)

            # acc_train_list.append(acc_train.cpu().numpy().tolist())
            # loss_train_list.append(loss_train)

            accDfTest = pd.DataFrame({'baseline':acc_test_list})
            accDfTest.to_csv("D:\\ChainsFLexps\\asynFL\\normal-10users\\AsynFL-idd{}-{}-{}localEpochs-{}users-{}Rounds_ACC_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

            lossDfTest = pd.DataFrame({'baseline':loss_test_list})
            lossDfTest.to_csv("D:\\ChainsFLexps\\asynFL\\normal-10users\\AsynFL-idd{}-{}-{}localEpochs-{}users-{}Rounds_Loss_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')


            # accDfTrain = pd.DataFrame({'baseline':acc_train_list})
            # accDfTrain.to_csv("D:\\ChainsFLexps\\asynFL\\AsynFL-Train-idd{}-{}-{}localEpochs-{}users-{}Rounds_ACC_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

            # lossDfTrain = pd.DataFrame({'baseline':loss_train_list})
            # lossDfTrain.to_csv("D:\\ChainsFLexps\\asynFL\\AsynFL-Train-idd{}-{}-{}localEpochs-{}users-{}Rounds_Loss_{}.csv".format(args.iid, args.model, args.local_ep, str(int(float(args.frac)*100)), args.epochs, dateNow),index=False,sep=',')

            print('# of real epoch is ' + str(realEpoch))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print("Testing loss: {:.2f}".format(loss_test))

            workerIterIdx[workerNow] += 1
            realEpoch += 1
