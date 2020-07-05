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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import buildModels


if __name__ == '__main__':
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()


    w_glob = net_glob.state_dict()


    loss_train = []
    
    # w_locals, loss_locals = [], []
    # m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    # for idx in idxs_users:
    #     local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    #     w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    #     w_locals.append(copy.deepcopy(w))
    #     loss_locals.append(copy.deepcopy(loss))
    # # update global weights
    # w_glob = FedAvg(w_locals)
    # torch.save(w_glob, '\\'+'parameter.pkl')
    # copy weight to net_glob


    # w_apv = []
    net_glob.load_state_dict(torch.load('../dagMainChain/clientS/paras/'+'aggModel-iter-1-epoch-1.pkl'))
    # tst = net_glob.state_dict()
    # w_apv.append(copy.deepcopy(tst))
    # print('The paras of tst are', tst)
    # net_glob.load_state_dict(torch.load('./data/paras/'+'9parameter.pkl'))
    # tst2 = net_glob.state_dict()
    # w_apv.append(copy.deepcopy(tst2))
    # print('The paras of tst2 are', tst2)
    # print('The combination of paras are', w_apv)
    # w_glob = FedAvg(w_apv)
    # print(w_glob)
    # net_glob.load_state_dict(w_glob)

    # print loss
    # iter = 1
    # loss_avg = sum(loss_locals) / len(loss_locals)
    # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    # loss_train.append(loss_avg)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

