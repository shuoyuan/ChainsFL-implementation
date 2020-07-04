#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import random
import time
from torchvision import datasets, transforms
import torch
from json import dumps
import os
import sys
import json
import buildModels

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':

    # shell envs
    shellEnv1 = "export PATH=${PWD}/../bin:$PATH"
    shellEnv2 = "export FABRIC_CFG_PATH=$PWD/../config/"
    shellEnv3 = "export CORE_PEER_TLS_ENABLED=true"
    shellEnv4 = "export CORE_PEER_LOCALMSPID=\"Org1MSP\""
    shellEnv5 = "export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
    shellEnv6 = "export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
    shellEnv7 = "export CORE_PEER_ADDRESS=localhost:7051"
    oneKeyEnv = shellEnv1 + " && " + shellEnv2 + " && " + shellEnv3 + " && " + shellEnv4 + " && " + shellEnv5 + " && " + shellEnv6 + " && " + shellEnv7

    # task release
    ## task info template {"Args":["set","taskRelease","{"taskID":"fl1234","epochs":10,"status":"start","usersFrac":0.1}"]}
    taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
    taskEpochs = args.epochs
    taskInitStatus = "start"
    taskUsersFrac = args.frac
    taskQuery = os.popen(r"./taskRelease.sh "+taskID+" "+str(taskEpochs)+" "+taskInitStatus+" "+str(taskUsersFrac))
    currentEpoch = 1

    # build network
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []

    # Device name list
    deviceName = []
    for i in range(args.num_users):
        deviceName.append("device"+("{:0>5d}".format(i)))

    while (currentEpoch <= args.epochs):
        flagList = copy.deepcopy(deviceName)
        w_locals, loss_locals = [], []
        while (len(flagList) != 0):
            for item in flagList:
                parasQueryShell = "peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"" + item +"\"]}'"
                parasQuery = os.popen(oneKeyEnv + " && " + parasQueryShell)
                parasInfoR = parasQuery.read()
                parasQuery.close()
                deviceUpdate = json.loads(parasInfoR)
                if deviceUpdate['taskID'] == taskID and deviceUpdate['epochs'] == currentEpoch:
                    localFileName = item + str(currentEpoch) + 'Epoch' + 'parameter.pkl'
                    getParasFile = os.popen('ipfs get ' + deviceUpdate['paras'] + ' ' + '-o ' + localFileName)
                    flagList.remove(item)
        for item in deviceName:
            w_locals.append(torch.load('./' + item + str(currentEpoch) + 'Epoch' + 'parameter.pkl'))
        w_glob = FedAvg(w_locals)
        globEpochFile = './' + str(currentEpoch) + 'Epoch' + 'parameter.pkl'
        torch.save(w_glob, globEpochFile)
        ## taskEpoch template {"Args":["set","taskID","{"epoch":1,"status":"runing","paras":"fileHash"}"]}
        if currentEpoch < args.epochs:
            currentStatus = "training"
        elif currentEpoch == args.epochs:
            currentStatus = "done"
        globParaInvoke = os.popen(r"./epochGlobUpdate.sh" + " " + globEpochFile + " " + taskID + " " + str(currentEpoch) + " " + currentStatus)
        currentEpoch += 1
        
    net_glob.load_state_dict(w_glob)

    # print loss
    iter = 1
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

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

