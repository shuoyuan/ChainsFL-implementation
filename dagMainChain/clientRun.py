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
import time
import threading
import shutil
import json
import random
import subprocess

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


# Number of tips confirmed by the new transaction
alpha = 2
## Number of tips needs to be kept greater than 3
beta = 3

nodeNum = 1

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

def main(aim_addr='127.0.0.1'):
    if os.path.exists('./clientS'):
        shutil.rmtree('./clientS')
    os.mkdir('./clientS')

    if os.path.exists('./clientS/paras'):
        shutil.rmtree('./clientS/paras')
    os.mkdir('./clientS/paras')

    # build model
    net_glob, args, dataset_train, dataset_test, dict_users = buildModels.modelBuild()
    net_glob.train()

    ## copy weights
    w_glob = net_glob.state_dict()

    # init the task ID
    taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
    iteration_count = 0

    # selected device
    ## init the list of device name
    allDeviceName = []
    for i in range(args.num_users):
        allDeviceName.append("device"+("{:0>5d}".format(i)))
    deviceSelected = []
    m = max(int(args.frac * args.num_users), 1) # args.frac is the fraction of users
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    for idx in idxs_users:
        deviceSelected.append(allDeviceName[idx])

    while 1:
        # Choose and require the apv trans
        apv_trans_name = []
        if iteration_count == 0:
            apv_trans_name.append('GenesisBlock')
        else:
            tips_list = 'tip_list'
            tips_file = './clientS/' + tips_list + '.json'
            dagClient.client_tips_require(aim_addr, tips_list, tips_file)
            with open(tips_file,'r') as f1:
                tips_dict = json.load(f1)
                f1.close()
            if len(tips_dict) <= alpha:
                apv_trans_name = list(tips_dict.keys())
            else:
                apv_trans_name = random.sample(tips_dict.keys(), alpha)

        print('\n******************')
        print('The approved tips are ', apv_trans_name)
        print('******************\n')

        # Get the trans file
        w_apv = []
        for apvTrans in apv_trans_name:
            apvTransFile =  './clientS/' + apvTrans + '.json'
            dagClient.client_trans_require(aim_addr, apvTrans, apvTransFile)
            print('This approved trans is ', apvTrans, ', and the file is ', apvTransFile)
            apvTransInfo = transaction.read_transaction(apvTransFile)
            apvParasFile = './clientS/paras/' + apvTrans + '.pkl'

            while 1:
                fileGetStatus, sttCodeGet = usefulTools.ipfsGetFile(apvTransInfo.model_para, apvParasFile)
                print('The filehash of this approved trans is ' + apvTransInfo.model_para + ', and the file is ' + apvParasFile + '!\n')
                if sttCodeGet == 0:
                    print(fileGetStatus)
                    print('\nThe apv parasfile ' + apvParasFile + ' has been downloaded!\n')
                    break
                else:
                    print(fileGetStatus)
                    print('\nFailed to download the apv parasfile ' + apvParasFile + ' !\n')

            # load the apv paras
            net_glob.load_state_dict(torch.load(apvParasFile, map_location=torch.device('cpu')))
            w_tmp = net_glob.state_dict()
            w_apv.append(copy.deepcopy(w_tmp))
        
        if len(w_apv) == 1:
            w_glob = w_apv[0]
        else:
            w_glob = FedAvg(w_apv)
        baseParasFile = './clientS/paras/baseModelParas-iter'+str(iteration_count)+'.pkl'
        torch.save(w_glob, baseParasFile)

        # Add the paras file of base model to ipfs network for sharding training
        while 1:
            basefileHash, baseSttCode = usefulTools.ipfsAddFile(baseParasFile)
            if baseSttCode == 0:
                print('\nThe base mode parasfile ' + baseParasFile + ' has been uploaded!\n')
                print('And the fileHash is ' + basefileHash + '\n')
                break
            else:
                print('Error: ' + basefileHash)
                print('\nFailed to uploaded the aggregated parasfile ' + baseParasFile + ' !\n')

        # Task release & model aggregation
        if iteration_count == 0:
            ## Task release
            taskEpochs = args.epochs
            taskInitStatus = "start"
            taskUsersFrac = args.frac
            while 1:
                taskRelease = subprocess.Popen(args=['../commonComponent/interRun.sh release '+taskID+' '+str(taskEpochs)+' '+taskInitStatus+' '+str(taskUsersFrac)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                trOuts, trErrs = taskRelease.communicate(timeout=10)
                if taskRelease.poll() == 0:
                    print(trOuts)
                    print('*** ' + taskID + ' has been released! ***\n')
                    break
                else:
                    print(trErrs)
                    print('*** Failed to release ' + taskID + ' ! ***\n')
                    time.sleep(2)

            ## Publish the initialization model
            ### taskEpoch template {"Args":["set","taskID","{"epoch":1,"status":"training","paras":"fileHash"}"]}
            while 1:
                spcAggModelPublish = subprocess.Popen(args=['../commonComponent/interRun.sh aggregated '+taskID+' 0 training '+basefileHash], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                aggPubOuts, aggPubErrs = spcAggModelPublish.communicate(timeout=10)
                if spcAggModelPublish.poll() == 0:
                    print(aggPubOuts)
                    print('*** The init aggModel of ' + taskID + ' has been published! ***\n')
                    break
                else:
                    print(aggPubErrs)
                    print('*** Failed to publish the init aggModel of ' + taskID + ' ! ***\n')
        else:
            currentEpoch = 1
            aggEchoFileHash = ''
            while (currentEpoch <= args.epochs):
                flagList = set(copy.deepcopy(deviceSelected))
                w_locals = []
                while (len(flagList) != 0):
                    flagSet = set()
                    lock = threading.Lock()
                    for deviceID in flagList:
                        t = threading.Thread(target=usefulTools.queryLocal,args=(lock,taskID,deviceID,currentEpoch,flagSet,))
                        t.start()
                        ts.append(t)
                    for t in ts:
                        t.join()
                    time.sleep(2)
                    flagList = flagList - flagSet
                for deviceID in flagSet:
                    localFileName = './clientS/paras/' + taskID + deviceID + 'Epoch' + str(currentEpoch) + '.pkl'
                    net_glob.load_state_dict(torch.load(localFileName))
                    tmpParas = net_glob.state_dict()
                    w_locals.append(copy.deepcopy(tmpParas))
                w_glob = FedAvg(w_locals)
                aggEchoParasFile = './clientS/paras/baseModelParas-iter'+str(iteration_count)+'.pkl'
                torch.save(w_glob, aggEchoParasFile)
                
                # aggEchoParasFile is the paras of this sharding trained in current epoch
                # Add the aggregated paras file to ipfs network
                while 1:
                    aggEchoFileHash, sttCodeAdd = usefulTools.ipfsAddFile(aggEchoParasFile)
                    if sttCodeAdd == 0:
                        print('\nThe aggregated parasfile ' + aggEchoParasFile + ' has been uploaded!\n')
                        print('And the fileHash is ' + aggEchoFileHash + '\n')
                        break
                    else:
                        print('Error: ' + aggEchoFileHash)
                        print('\nFailed to uploaded the aggregated parasfile ' + aggEchoParasFile + ' !\n')

                ## Publish the aggregated model paras trained in this epoch
                ### taskEpoch template {"Args":["set","taskID","{"epoch":1,"status":"training","paras":"fileHash"}"]}
                while 1:
                    epochAggModelPublish = subprocess.Popen(args=['../commonComponent/interRun.sh aggregated '+taskID+' '+str(currentEpoch)+' training '+aggEchoFileHash], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                    aggPubOuts, aggPubErrs = epochAggModelPublish.communicate(timeout=10)
                    if epochAggModelPublish.poll() == 0:
                        print(aggPubOuts)
                        print('*** The aggModel of ' + taskID + ' has been published! ***\n')
                        break
                    else:
                        print(aggPubErrs)
                        print('*** Failed to publish the init aggModel of ' + taskID + ' ! ***\n')
                currentEpoch += 1
        
            new_trans = transaction.Transaction(time.time(), nodeNum,'', aggEchoFileHash, apv_trans_name)

            # upload the trans to DAG network
            dagClient.trans_upload(aim_addr, new_trans)

            print('\n******************')
            print('The details of this trans are', new_trans)
            print('******************\n')
            print('*** The trans generated in the iteration #%d had been uploaded!'%iteration_count+' ***\n')
            print('*************************************************************************************\n')
        iteration_count += 1
        time.sleep(10)

if __name__ == '__main__':
    main('127.0.0.1')