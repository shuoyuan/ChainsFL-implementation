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


# The number of tips confirmed by the new transaction
alpha = 2
## The number of tips needs to be kept greater than 3
beta = 3

nodeNum = 1

def main(aim_addr='127.0.0.1'):
    if os.path.exists('./clientS'):
        shutil.rmtree('./clientS')
    os.mkdir('./clientS')

    if os.path.exists('./clientS/paras'):
        shutil.rmtree('./clientS/paras')
    os.mkdir('./clientS/paras')

    with open('./run_state.txt','w') as f:
        f.write('-1')
        f.close()

    print('Detect status of the task')
    # while 1:
    #     with open('./run_state.txt', 'r') as f:
    #         state = f.read()
    #         if state != '-1':
    #             f.close()
    #             break
    #     time.sleep(1)
    #     print('Task not started!')

    # print('Task is runing!')

    # with open('./run_state.txt', 'r') as f:
    #     rank = int(f.read())
    # print('The epoch of this sharding is ' + str(rank))

    iteration_count = 0

    while 1:
        # build model
        args = args_parser()
        # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        args.device = torch.device('cpu')

        ## load dataset and split users
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

        ## copy weights
        w_glob = net_glob.state_dict()
        w_apv = []

        # Choose and require the apv trans
        iteration_count += 1
        apv_trans_name = []
        if iteration_count == 1:
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
        for apvTrans in apv_trans_name:
            apvTransFile =  './clientS/' + apvTrans + '.json'
            dagClient.client_trans_require(aim_addr, apvTrans, apvTransFile)
            print('This approved trans is ', apvTrans, ', and the file is ', apvTransFile)
            apvTransInfo = transaction.read_transaction(apvTransFile)
            apvParasFile = './clientS/paras/' + apvTrans + 'pkl'

            while 1:
                fileGetStatus, sttCodeGet = usefulTools.ipfsGetFile(apvTransInfo.model_para, apvParasFile)
                if sttCodeGet == 0:
                    print(fileGetStatus)
                    print('\nThe apv parasfile ' + apvParasFile + ' has been downloaded!\n')
                    break
                else:
                    print('\nFailed to download the apv parasfile ' + apvParasFile + ' !\n')
            print('The filehash of this approved trans is ' + apvTransInfo.model_para + ', and the file is ' + apvParasFile + '!\n')

            # load the apv paras
            net_glob.load_state_dict(torch.load(apvParasFile))
            w_tmp = net_glob.state_dict()
            w_apv.append(copy.deepcopy(w_tmp))
        
        if len(w_apv) == 1:
            w_glob = w_apv[0]
        else:
            w_glob = FedAvg(w_apv)
        aggParasFile = './clientS/paras/agg-'+str(iteration_count)+'parameter.pkl'
        torch.save(w_glob, aggParasFile)

        # Add the aggregated paras file to ipfs network
        while 1:
            fileHash, sttCodeAdd = usefulTools.ipfsAddFile(aggParasFile)
            if sttCodeAdd == 0:
                print('\nThe aggregated parasfile ' + aggParasFile + ' has been uploaded!\n')
                break
            else:
                print('\nFailed to uploaded the aggregated parasfile ' + aggParasFile + ' !\n')
        new_trans = transaction.Transaction(time.time(), nodeNum,'', fileHash, apv_trans_name)
        dagClient.trans_upload(aim_addr, new_trans)
        print('\n******************')
        print('The details of this trans are', new_trans)
        print('******************\n')
        print('*** The trans generated in the iteration #%d had been uploaded!'%iteration_count+' ***\n')
        print('*************************************************************************************\n')
        time.sleep(10)

if __name__ == '__main__':
    main('127.0.0.1')