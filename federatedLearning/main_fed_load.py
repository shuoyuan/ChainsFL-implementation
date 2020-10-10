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
import os
from json import dumps

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import buildModels
import pickle


if __name__ == '__main__':
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()
    print(dict_users)
    with open('../commonComponent/noniid_dict_users.pkl', 'wb') as f:
        pickle.dump(dict_users, f)
    w_glob = net_glob.state_dict()

    # # w_apv = []
    # # fileName = os.listdir("D:\\test")
    # # for item in fileName:
    # #     net_glob.load_state_dict(torch.load('D:\\test\\'+item))
    # #     w_apv.append(copy.deepcopy(net_glob.state_dict()))
    # # w_glob2 = FedAvg(w_apv)
    # # net_glob.load_state_dict(w_glob2)
    # torch.save(w_glob, "D:\\genesis.pkl")
    # # tst_glob = torch.load('D:\\Documents\\blockchain\\federatedLearningBC\\federatedLearning\\data\\paras\\9parameter.pkl')
    # # print(tst_glob)
    # # net_glob.load_state_dict(tst_glob)

    # # plot loss curve
    # # plt.figure()
    # # plt.plot(range(len(loss_train)), loss_train)
    # # plt.ylabel('train_loss')
    # # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # # testing
    # net_glob.eval()
    # # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # # print("Training accuracy: {:.2f}".format(acc_train))
    # # print("Testing accuracy: {:.2f}".format(acc_test))
    # print(acc_test.cpu().numpy().tolist())

