#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import time
from torchvision import datasets, transforms
import torch
from json import dumps
import os
import sys
import json

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

    # query task release info
    ## task info template {"Args":["set","taskRelease","{"taskID":"fl1234","epochs":10,"status":"start","usersFrac":0.1}"]}
    taskQueryshell = "peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"taskRelease\"]}'"
    taskQuery = os.popen(oneKeyEnv + " && " + taskQueryshell)
    taskInfoR = taskQuery.read()
    taskQuery.close()
    taskInfo = json.loads(taskInfoR)
    args.frac = taskInfo['usersFrac']
    args.epochs = taskInfo['epochs']
    taskID = taskInfo['taskID']

    # query specific task status
    taskSttQueryshell = "peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"" + taskID + "\"]}'"
    ## !! need to check the runing status
    taskSttQuery = os.popen(oneKeyEnv + " && " + taskSttQueryshell)
    taskRT = taskSttQuery.read()
    taskSttQuery.close()
    taskCurStt = json.loads(taskRT)
    taskStatus = taskCurStt['status'] + 1

    if taskStatus == 'training':
        currentEpoch = 1
    else:
        currentEpoch = taskCurStt['epoch']

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

    while (taskStatus != "done"):
        if currentEpoch > 1 and currentEpoch < args.epochs:
            ## taskEpoch template {"Args":["set","taskID","{"epoch":1,"status":"training","paras":"fileHash"}"]}
            taskSttQueryshell = "peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"" + taskID + "\"]}'"
            ## !! need to check the runing status
            taskSttQuery = os.popen(oneKeyEnv + " && " + taskSttQueryshell)
            taskRT = taskSttQuery.read()
            taskSttQuery.close()
            taskCurStt = json.loads(taskRT)
            taskStatus = taskCurStt['status']

            if (int(currentEpoch) - 1) == taskCurStt['epoch']:
                getaggFile = os.popen('ipfs get ' + taskCurStt['fileHash'] + ' ' + '-o ' + str(currentEpoch-1) + 'Epoch' + 'parameter.pkl')
                net_glob.load_state_dict(torch.load('./'+str(currentEpoch-1)+'Epoch'+'parameter.pkl'))
                
                currentEpoch += 1
                w_locals, loss_locals = [], []
                m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                for idx in idxs_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    torch.save(w, './'+'epoch'+str(currentEpoch-1)+deviceName[idx]+'parameter.pkl')
                    saccInvoke = os.popen(r"./invokeParameters.sh" + " " + localFileName + " " + deviceName[idx]  + " " + taskID + " " + str(currentEpoch-1))
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
            else:
                time.sleep(2)
        else: 
            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                localFileName = './'+'epoch'+str(currentEpoch)+deviceName[idx]+'parameter.pkl'
                torch.save(w, localFileName)
                saccInvoke = os.popen(r"./invokeParameters.sh" + " " + localFileName + " " + deviceName[idx]  + " " + taskID + " " + str(currentEpoch))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                currentEpoch += 1
    # update global weights
    w_glob = FedAvg(w_locals)
    # torch.save(w_glob, '\\'+'parameter.pkl')
    # copy weight to net_glob
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

