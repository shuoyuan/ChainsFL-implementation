# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : clientRun.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-18 21:10
# Description  : 
# ******************************************************

import sys
from dagComps import transaction
import socket
import os
from dagSocket import dagClient
from torch.utils.tensorboard import SummaryWriter
import time
import threading
import shutil
import json
import random
import subprocess
import pickle
import pandas as pd

# Common Components
sys.path.append('../commonComponent')
import usefulTools

## FL related
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

sys.path.append('../federatedLearning')
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import buildModels
import datetime


# Number of tips selected by the leader of shard blockchain
alpha = 3
# Number of tips needs to be kept greater than 3
beta = 3
# Number of tips confirmed by the new transaction
gamma = 2

# Index of shard network
nodeNum = 1
# Rounds trained in shard
# shard_round = 4

# shell envs of Org1
fabricLocation = "export FabricL=/home/shawn/Documents/fabric-samples/test-network"
shellEnv1 = "export PATH=${FabricL}/../bin:$PATH"
shellEnv2 = "export FABRIC_CFG_PATH=${FabricL}/../config/"
shellEnv3 = "export CORE_PEER_TLS_ENABLED=true"
shellEnv4 = "export CORE_PEER_LOCALMSPID=\"Org1MSP\""
shellEnv5 = "export CORE_PEER_TLS_ROOTCERT_FILE=${FabricL}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
shellEnv6 = "export CORE_PEER_MSPCONFIGPATH=${FabricL}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
shellEnv7 = "export CORE_PEER_ADDRESS=localhost:7051"
oneKeyEnv = shellEnv1 + " && " + shellEnv2 + " && " + shellEnv3 + " && " + shellEnv4 + " && " + shellEnv5 + " && " + shellEnv6 + " && " + shellEnv7

# Acc and Loss of the model trained by shard
nodeTestAcc = []
nodeTestLoss = []

# Acc and Loss on training set
nodeTrainAcc = []
nodeTrainLoss = []

def main(aim_addr='149.129.40.183'):
    if os.path.exists('./clientS'):
        shutil.rmtree('./clientS')
    os.mkdir('./clientS')

    if os.path.exists('./clientS/paras'):
        shutil.rmtree('./clientS/paras')
    os.mkdir('./clientS/paras')

    if os.path.exists('./clientS/paras/apvTrans'):
        shutil.rmtree('./clientS/paras/apvTrans')
    os.mkdir('./clientS/paras/apvTrans')

    if os.path.exists('./clientS/paras/local'):
        shutil.rmtree('./clientS/paras/local')
    os.mkdir('./clientS/paras/local')

    if os.path.exists('./clientS/tipsJson'):
        shutil.rmtree('./clientS/tipsJson')
    os.mkdir('./clientS/tipsJson')

    if os.path.exists('./clientS/apvJson'):
        shutil.rmtree('./clientS/apvJson')
    os.mkdir('./clientS/apvJson')


    # build model
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    ## copy weights
    w_glob = net_glob.state_dict()

    iteration_count = 0

    # selected device
    ## init the list of device name
    allDeviceName = []
    for i in range(args.num_users):
        allDeviceName.append("device"+("{:0>5d}".format(i)))
    deviceSelected = []

    # Randomly select the devices

    # m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # print('\n**************************** Idxs of selected devices *****************************')
    # print('The idxs of selected devices are\n', idxs_users)
    # print('*************************************************************************************\n')

    # ## Exchange the info of selected device with fabric
    # with open('../commonComponent/selectedDeviceIdxs.txt', 'wb') as f:
    #     pickle.dump(idxs_users, f)

    idxs_users = [ 5, 56, 76, 78, 68, 25, 47, 15, 61, 55]
    # idxs_users = [60, 37, 27, 70, 79, 34, 18, 88, 57, 98]
    # idxs_users = [48, 46, 33, 82,  4,  7,  6, 91, 92, 52]

    dateNow = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    basic_acc_list = []


    ## tensorboard Part

    # writer = SummaryWriter()
    # writer.add_scalars('Acc', {'Acc_with_Check': 0}, 0)
    # tenfig_data = []

    # writer.add_scalars('Acc', {'Acc_without_Check': 0}, 0)
    # AWOC_fileName = '/root/shard3/data/shard_3_round3_tenfig_data_cnn_20200821135421.csv'
    # acc_wo_check_csv = pd.read_csv(AWOC_fileName)
    # acc_wo_check_data = np.array(acc_wo_check_csv['shard_3'])


    # writer.add_scalars('Loss', {'Loss_with_Check': 1}, 0)
    # tenfig_data_loss = []

    # writer.add_scalars('Loss', {'Loss_without_Check': 1}, 0)
    # LWOC_fileName = '/root/shard3/data/shard_3_round3_tenfig_data_loss_cnn_20200821135421.csv'
    # loss_wo_check_csv = pd.read_csv(LWOC_fileName)
    # loss_wo_check_data = np.array(loss_wo_check_csv['shard_3'])[1:]

    # tensor_iter = 1

    ## tensorboard part
    

    ## Exchange the info of selected device with fabric: dict_user_fileHash:QmTuvPRDGnLm95fL7uxxhvegoWL3Q9YyAUtEsTK5ZetN4W
    dict_users_file = "../commonComponent/dict_users.pkl"
    dict_userf_fileHash = "QmTuvPRDGnLm95fL7uxxhvegoWL3Q9YyAUtEsTK5ZetN4W"
    while 1:
        dictUserGetStatus, dictUsersttCodeGet = usefulTools.ipfsGetFile(dict_userf_fileHash, dict_users_file)
        print('The filehash of this dict_user is ' + dict_userf_fileHash + ' , and the file is ' + dict_users_file + '!')
        if dictUsersttCodeGet == 0:
            print(dictUserGetStatus.strip())
            print('The dict_user file ' + dict_users_file + ' has been downloaded!\n')
            break
        else:
            print(dictUserGetStatus)
            print('\nFailed to download the dict_user file  ' + dict_users_file + ' !\n')
    
    with open(dict_users_file, 'rb') as f:
        dict_users = pickle.load(f)

    for idx in idxs_users:
        deviceSelected.append(allDeviceName[idx])
    print('\n**************************** Selected devices *****************************')
    print('The idxs of selected devices are\n', deviceSelected)
    print('*****************************************************************************\n')
    while 1:
        print('\n\n\n**************************** Iteration %d *****************************'%iteration_count)
        # init the task ID
        taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))

        # Choose and require the apv trans
        apv_trans_cands = []
        if iteration_count == 0:
            apv_trans_cands.append('GenesisBlock')
        else:
            tips_list = 'tip_list'
            tips_file = './clientS/tipsJson/iteration-' + str(iteration_count) + '-' + tips_list + '.json'
            dagClient.client_tips_require(aim_addr, tips_list, tips_file)
            ## try to fix the JSONDecodeError
            try:
                with open(tips_file, encoding='utf-8-sig', errors='ignore', mode='r') as f1:
                    tips_dict = json.load(f1)
                    f1.close()
            except:
                time.sleep(2)
                dagClient.client_tips_require(aim_addr, tips_list, tips_file)
                with open(tips_file, encoding='utf-8-sig', errors='ignore', mode='r') as f1:
                    tips_dict = json.load(f1)
                    f1.close()

            if len(tips_dict) <= alpha:
                apv_trans_cands = list(tips_dict.keys())
            else:
                apv_trans_cands = random.sample(tips_dict.keys(), alpha)

        print('\n************************* Select Candidates Tips *****************************')
        print('The candidates tips are ', apv_trans_cands)
        print('********************************************************************************\n')

        # Get the trans file and the model paras file
        apv_trans_cands_dict = {}
        for apvTrans in apv_trans_cands:
            apvTransFile =  './clientS/apvJson/' + apvTrans + '.json'
            dagClient.client_trans_require(aim_addr, apvTrans, apvTransFile)
            print('\nThis approved trans is ', apvTrans, ', and the file is ', apvTransFile)
            apvTransInfo = transaction.read_transaction(apvTransFile)
            apvParasFile = './clientS/paras/apvTrans/iteration-' + str(iteration_count) + '-' + apvTrans + '.pkl'

            while 1:
                fileGetStatus, sttCodeGet = usefulTools.ipfsGetFile(apvTransInfo.model_para, apvParasFile)
                print('The filehash of this approved trans is ' + apvTransInfo.model_para + ', and the file is ' + apvParasFile + '!')
                if sttCodeGet == 0:
                    print(fileGetStatus.strip())
                    print('The apv parasfile ' + apvParasFile + ' has been downloaded!\n')
                    break
                else:
                    print(fileGetStatus)
                    print('\nFailed to download the apv parasfile ' + apvParasFile + ' !\n')
            apv_trans_cands_dict[apvTransInfo.name] = float(apvTransInfo.model_acc)

        # select tips for aggregation of basic model     !!! key function
        apv_trans_final = []
        if len(apv_trans_cands_dict) == alpha:
            sort_dict = sorted(apv_trans_cands_dict.items(),key=lambda x:x[1],reverse=True)
            for i in range(gamma):
                apv_trans_final.append(sort_dict[i][0])
        else:
            apv_trans_final = apv_trans_cands

        # load the apv paras
        w_apv = []
        for item in apv_trans_final:
            apvParasFile = './clientS/paras/apvTrans/iteration-' + str(iteration_count) + '-' + item + '.pkl'
            net_glob.load_state_dict(torch.load(apvParasFile, map_location=torch.device('cpu')))
            w_tmp = net_glob.state_dict()
            w_apv.append(copy.deepcopy(w_tmp))
        
        if len(w_apv) == 1:
            w_glob = w_apv[0]
        else:
            w_glob = FedAvg(w_apv)
        baseParasFile = './clientS/paras/baseModelParas-iter'+str(iteration_count)+'.pkl'
        torch.save(w_glob, baseParasFile)

        # evalute the acc of basic model obtain from DAG
        basicModelAcc, basicModelLoss = buildModels.evalua(net_glob, w_glob, dataset_test, args)
        basicModelAcc = basicModelAcc.cpu().numpy().tolist()
        
        print("\n************************************")
        print("Acc of the basic model in iteration "+str(iteration_count)+" is "+str(basicModelAcc))
        print("************************************")

        basic_acc_list.append(basicModelAcc)
        basicAccDf = pd.DataFrame({'shard_{}'.format(nodeNum):basic_acc_list})
        basicAccDf.to_csv("../data/shard_{}_round{}_basic_acc_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')

        # Add the paras file of base model to ipfs network for shard training
        while 1:
            basefileHash, baseSttCode = usefulTools.ipfsAddFile(baseParasFile)
            if baseSttCode == 0:
                print('\nThe base mode parasfile ' + baseParasFile + ' has been uploaded!')
                print('And the fileHash is ' + basefileHash + '\n')
                break
            else:
                print('Error: ' + basefileHash)
                print('\nFailed to uploaded the aggregated parasfile ' + baseParasFile + ' !\n')

        # Task release & model aggregation
        ## Task release
        taskEpochs = args.epochs
        taskInitStatus = "start"
        taskUsersFrac = args.frac
        while 1:
            taskRelease = subprocess.Popen(args=['../commonComponent/interRun.sh release '+taskID+' '+str(taskEpochs)+' '+taskInitStatus+' '+str(taskUsersFrac)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            trOuts, trErrs = taskRelease.communicate(timeout=10)
            if taskRelease.poll() == 0:
                print('*** ' + taskID + ' has been released! ***')
                print('*** And the detail of this task is ' + trOuts.strip() + '! ***\n')
                break
            else:
                print(trErrs)
                print('*** Failed to release ' + taskID + ' ! ***\n')
                time.sleep(2)

        ## Publish the init base model
        ### taskEpoch template {"Args":["set","taskID","{"epoch":1,"status":"training","paras":"fileHash"}"]}
        while 1:
            spcAggModelPublish = subprocess.Popen(args=['../commonComponent/interRun.sh aggregated '+taskID+' 0 training '+basefileHash], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            aggPubOuts, aggPubErrs = spcAggModelPublish.communicate(timeout=10)
            if spcAggModelPublish.poll() == 0:
                print('*** The init aggModel of ' + taskID + ' has been published! ***')
                print('*** And the detail of the init aggModel is ' + aggPubOuts.strip() + ' ! ***\n')
                break
            else:
                print(aggPubErrs)
                print('*** Failed to publish the init aggModel of ' + taskID + ' ! ***\n')
        
        ## wait the local train
        time.sleep(10)
        
        ## Aggregated the local model trained by the selected devices
        currentEpoch = 1
        aggEchoFileHash = ''
        aggModelAcc = 50.0
        while (currentEpoch <= args.epochs):
            flagList = set(copy.deepcopy(deviceSelected))
            w_locals = []
            while (len(flagList) != 0):
                flagSet = set()
                ts = []
                lock = threading.Lock()
                for deviceID in flagList:
                    localFileName = './clientS/paras/local/' + taskID + '-' + deviceID + '-epoch-' + str(currentEpoch) + '.pkl'
                    t = threading.Thread(target=usefulTools.queryLocal,args=(lock,taskID,deviceID,currentEpoch,flagSet,localFileName,))
                    t.start()
                    ts.append(t)
                for t in ts:
                    t.join()
                time.sleep(2)
                flagList = flagList - flagSet
            for deviceID in deviceSelected:
                localFileName = './clientS/paras/local/' + taskID + '-' + deviceID + '-epoch-' + str(currentEpoch) + '.pkl'

                ## no check
                # net_glob.load_state_dict(torch.load(localFileName))
                # tmpParas = net_glob.state_dict()
                # w_locals.append(copy.deepcopy(tmpParas))
                
                ## check the acc of the models trained by selected device & drop the low quality model
                canddts_dev_pas = torch.load(localFileName,map_location=torch.device('cpu'))
                acc_canddts_dev, loss_canddts_dev = buildModels.evalua(net_glob, canddts_dev_pas, dataset_test, args)
                acc_canddts_dev = acc_canddts_dev.cpu().numpy().tolist()
                print("Test acc of the model trained by "+str(deviceID)+" is " + str(acc_canddts_dev))
                if (acc_canddts_dev - aggModelAcc) < -10:
                    print(str(deviceID)+" is a malicious device!")
                else:
                    w_locals.append(copy.deepcopy(canddts_dev_pas))
                    
            w_glob = FedAvg(w_locals)
            aggEchoParasFile = './clientS/paras/aggModel-iter-'+str(iteration_count)+'-epoch-'+str(currentEpoch)+'.pkl'
            torch.save(w_glob, aggEchoParasFile)

            # evalute the acc of datatest
            aggModelAcc, aggModelLoss = buildModels.evalua(net_glob, w_glob, dataset_test, args)
            aggModelAcc = aggModelAcc.cpu().numpy().tolist()


            # save the acc and loss
            nodeTestAcc.append(aggModelAcc)
            nodeTestLoss.append(aggModelLoss)

            nodeTestAccDf = pd.DataFrame({'shard_{}'.format(nodeNum):nodeTestAcc})
            nodeTestAccDf.to_csv("../data/result/shard_{}_round{}_test_acc_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')

            nodeTestLossDf = pd.DataFrame({'shard_{}'.format(nodeNum):nodeTestLoss})
            nodeTestLossDf.to_csv("../data/result/shard_{}_round{}_test_loss_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')
            

            # evalute the acc on training set
            aggModelTrainAcc, aggModelTrainLoss = buildModels.evalua(net_glob, w_glob, dataset_train, args)
            aggModelTrainAcc = (aggModelTrainAcc.cpu().numpy().tolist())/100

            # save the acc and loss on training set
            nodeTrainAcc.append(aggModelTrainAcc)
            nodeTrainLoss.append(aggModelTrainLoss)

            nodeTrainAccDf = pd.DataFrame({'shard_{}'.format(nodeNum):nodeTrainAcc})
            nodeTrainAccDf.to_csv("../data/result/shard_{}_round{}_train_acc_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')

            nodeTrainLossDf = pd.DataFrame({'shard_{}'.format(nodeNum):nodeTrainLoss})
            nodeTrainLossDf.to_csv("../data/result/shard_{}_round{}_train_loss_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')

            # add the tensorboard view

            # writer.add_scalars('Acc', {'Acc_with_Check': aggModelAcc/100}, tensor_iter)
            # writer.add_scalars('Acc', {'Acc_without_Check': acc_wo_check_data[tensor_iter-1]}, tensor_iter)
            # tenfig_data.append(aggModelAcc/100)
            # tenfig_data_df = pd.DataFrame({'shard_{}'.format(nodeNum):tenfig_data})
            # tenfig_data_df.to_csv("../data/shard_{}_round{}_tenfig_data_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')

            # writer.add_scalars('Loss', {'Loss_with_Check': aggModelLoss}, tensor_iter)
            # writer.add_scalars('Loss', {'Loss_without_Check': loss_wo_check_data[tensor_iter-1]}, tensor_iter)
            # tenfig_data_loss.append(aggModelLoss)
            # tenfig_data_loss_df = pd.DataFrame({'shard_{}'.format(nodeNum):tenfig_data_loss})
            # tenfig_data_loss_df.to_csv("../data/shard_{}_round{}_tenfig_data_loss_{}_{}.csv".format(nodeNum, args.epochs, args.model, dateNow),index=False,sep=',')
            
            # tensor_iter += 1

            print("\n************************************")
            print("Acc of the agg model of Round "+str(currentEpoch)+" in iteration "+str(iteration_count)+" is "+str(aggModelAcc))
            print("************************************")
            
            # aggEchoParasFile is the paras of this sharding trained in current epoch
            # Add the aggregated paras file to ipfs network
            while 1:
                aggEchoFileHash, sttCodeAdd = usefulTools.ipfsAddFile(aggEchoParasFile)
                if sttCodeAdd == 0:
                    print('\n*************************')
                    print('The aggregated parasfile ' + aggEchoParasFile + ' has been uploaded!')
                    print('And the fileHash is ' + aggEchoFileHash + '!')
                    print('*************************\n')
                    break
                else:
                    print('Error: ' + aggEchoFileHash)
                    print('\nFailed to uploaded the aggregated parasfile ' + aggEchoParasFile + ' !\n')

            ## Publish the aggregated model paras trained in this epoch
            ### taskEpoch template {"Args":["set","taskID","{"epoch":1,"status":"training","paras":"fileHash"}"]}
            taskStatus = 'training'
            if currentEpoch == args.epochs:
                taskStatus = 'done'
            while 1:
                epochAggModelPublish = subprocess.Popen(args=['../commonComponent/interRun.sh aggregated '+taskID+' '+str(currentEpoch)+' '+taskStatus+' '+aggEchoFileHash], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                aggPubOuts, aggPubErrs = epochAggModelPublish.communicate(timeout=10)
                if epochAggModelPublish.poll() == 0:
                    print('\n******************')
                    print('The info of task ' + taskID + ' is ' + aggPubOuts.strip())
                    print('The model aggregated in epoch ' + str(currentEpoch) + ' for ' + taskID + ' has been published!')
                    print('******************\n')
                    break
                else:
                    print(aggPubErrs)
                    print('*** Failed to publish the Model aggregated in epoch ' + str(currentEpoch) + ' for ' + taskID + ' ! ***\n')
            currentEpoch += 1
        
        new_trans = transaction.Transaction(time.time(), nodeNum, aggModelAcc, aggEchoFileHash, apv_trans_final)

        # upload the trans to DAG network
        dagClient.trans_upload(aim_addr, new_trans)

        print('\n******************************* Transaction upload *******************************')
        print('The details of this trans are', new_trans)
        print('The trans generated in the iteration #%d had been uploaded!'%iteration_count)
        print('*************************************************************************************\n')
        iteration_count += 1
        time.sleep(2)
    writer.close()

if __name__ == '__main__':
    main('192.168.2.12')