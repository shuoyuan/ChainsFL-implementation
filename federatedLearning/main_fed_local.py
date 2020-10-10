# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : main_fed_local.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-02 16:08
# Description  : 
# ******************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import time
import shutil
from torchvision import datasets, transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from json import dumps
import os
import sys
import json
import pickle
import subprocess
import datetime

# Common Components
sys.path.append('../commonComponent')
import usefulTools

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import buildModels


if __name__ == '__main__':

    if os.path.exists('./data/local'):
        shutil.rmtree('./data/local')
    os.mkdir('./data/local')

    if os.path.exists('./data/paras'):
        shutil.rmtree('./data/paras')
    os.mkdir('./data/paras')

    # build network
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    with open('../commonComponent/dict_users.pkl', 'rb') as f:
        dict_users = pickle.load(f)
    
    ## Used to check whether it is a new task
    checkTaskID = ''

    iteration = 0
    while 1:
        taskRelInfo = {}
        # taskRelease info template {"taskID":"task1994","epoch":10,"status":"start","usersFrac":0.1}
        while 1:
            taskRelQue, taskRelQueStt = usefulTools.simpleQuery('taskRelease')
            if taskRelQueStt == 0:
                taskRelInfo = json.loads(taskRelQue)
                print('\n*************************************************************************************')
                print('Latest task release status is %s!'%taskRelQue.strip())
                break
        taskRelEpoch = int(taskRelInfo['epoch'])
        
        # task info template {"epoch":"0","status":training,"paras":"QmSaAhKsxELzzT1uTtnuBacgfRjhehkzz3ybwnrkhJDbZ2"}
        taskID = taskRelInfo['taskID']
        print('Current task is',taskID)
        taskInfo = {}
        while 1:
            taskInQue, taskInQueStt = usefulTools.simpleQuery(taskID)
            if taskInQueStt == 0:
                taskInfo = json.loads(taskInQue)
                print('Latest task info is %s!'%taskInQue.strip())
                print('*************************************************************************************\n')
                break
        if taskInfo['status'] == 'done' or checkTaskID == taskID:
            print('*** %s has been completed! ***\n'%taskID)
            time.sleep(5)
        else:
            print('\n******************************* Iteration #%d starting ********************************'%iteration+'\n')
            print('Iteration %d starting!'%iteration)
            print('\n*************************************************************************************\n')
            currentEpoch = int(taskInfo['epoch']) + 1
            loss_train = []
            while currentEpoch <= taskRelEpoch:
                ## query the task info of current epoch
                while 1:
                    taskInQueEpo, taskInQueEpoStt = usefulTools.simpleQuery(taskID)
                    if taskInQueEpoStt == 0:
                        taskInfoEpo = json.loads(taskInQueEpo)
                        if int(taskInfoEpo['epoch']) == (currentEpoch-1):
                            print('\n****************************** Latest status of %s ******************************'%taskID)
                            print('(In loop) Latest task info is \n %s!'%taskInQueEpo)
                            print('*************************************************************************************\n')
                            break
                        else:
                            print('\n*************************** %s has not been updated ***************************'%taskID)
                            print('(In loop) Latest task info is \n %s!'%taskInQueEpo)
                            print('*************************************************************************************\n')
                            time.sleep(10)
                ## download the paras file of aggregated model for training in current epoch 
                aggBasModFil = './data/paras/aggModel-iter-' + str(iteration) + '-epoch-' + str(currentEpoch-1) + '.pkl'
                while 1:
                    aggBasMod, aggBasModStt = usefulTools.ipfsGetFile(taskInfoEpo['paras'], aggBasModFil)
                    if aggBasModStt == 0:
                        print('\nThe paras file of aggregated model for epoch %d training has been downloaded!\n'%(int(taskInfoEpo['epoch'])+1))
                        break
                    else:
                        print('\nFailed to download the paras file of aggregated model for epoch %d training!\n'%(int(taskInfoEpo['epoch'])+1))

                # copy weights and load the base model paras
                w_glob = net_glob.state_dict()
                net_glob.load_state_dict(torch.load(aggBasModFil))

                # Device name list
                # with open('../commonComponent/selectedDeviceIdxs.txt', 'rb') as f:
                #     idxs_users = pickle.load(f)

                idxs_users = [ 5, 56, 76, 78, 68, 25, 47, 15, 61, 55]

                ## init the list of device name
                allDeviceName = []
                for i in range(args.num_users):
                    allDeviceName.append("device"+("{:0>5d}".format(i)))

                # print('\n**************************** All devices *****************************')
                # print(allDeviceName)
                # print('*************************************************************************************\n')

                loss_locals = []
                for idx in idxs_users:
                    if idx not in []:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                        loss_locals.append(copy.deepcopy(loss))
                        devLocFile = './data/local/' + taskID + '-' + allDeviceName[idx] + '-epoch-' + str(currentEpoch) + '.pkl'
                        torch.save(w, devLocFile)
                        while 1:
                            localAdd, localAddStt = usefulTools.ipfsAddFile(devLocFile)
                            if localAddStt == 0:
                                print('%s has been added to the IPFS network!'%devLocFile)
                                print('And the hash value of this file is %s'%localAdd)
                                break
                            else:
                                print('Failed to add %s to the IPFS network!'%devLocFile)
                    else:
                        # For CNN
                        localAdd = "QmTZqGKUEvD5F8vQyEEJLJB7rzX17tpnN2Uu4YWBRZEYQx"
                        # For MLP
                        #localAdd = "QmaBYCmzPQ2emuXpVykLDHra7t8tPiU8reFMkbHpN1rRoo"
                    while 1:
                        localRelease = subprocess.Popen(args=['../commonComponent/interRun.sh local '+allDeviceName[idx]+' '+taskID+' '+str(currentEpoch)+' '+localAdd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                        localOuts, localErrs = localRelease.communicate(timeout=10)
                        if localRelease.poll() == 0:
                            print('*** Local model train in epoch ' + str(currentEpoch) + ' of ' + allDeviceName[idx] + ' has been uploaded! ***\n')
                            break
                        else:
                            print(localErrs.strip())
                            print('*** Failed to release Local model train in epoch ' + str(currentEpoch) + ' of ' + allDeviceName[idx] + '! ***\n')
                            time.sleep(2)

                loss_avg = sum(loss_locals) / len(loss_locals)
                print('Epoch {:3d}, Average loss {:.3f}'.format(currentEpoch, loss_avg))
                loss_train.append(loss_avg)
                currentEpoch += 1

            checkTaskID = taskID
            # plot loss curve
            plt.figure()
            plt.plot(range(len(loss_train)), loss_train)
            plt.ylabel('train_loss')
            plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_iteration{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, iteration, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
            print('\n*********************************** Iteration #%d ***********************************'%iteration+'\n')
            print('Current iteration %d has been completed!'%iteration)
            print('\n*************************************************************************************\n')
            iteration += 1
                
            # testing
            # net_glob.eval()
            # acc_train, loss_train = test_img(net_glob, dataset_train, args)
            # acc_test, loss_test = test_img(net_glob, dataset_test, args)
            # print("Training accuracy: {:.2f}".format(acc_train))
            # print("Testing accuracy: {:.2f}".format(acc_test))

