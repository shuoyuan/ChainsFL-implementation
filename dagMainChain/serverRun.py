# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : serverRun.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-18 20:05
# Description  : 
# ******************************************************

import sys
from dagComps import transaction
from dagComps.dag import DAG
from dagSocket import dagServer
import socket
import os
import time
import shutil
import numpy as np
import time
import glob

# The number of tips confirmed by the new transaction
alpha = 2
## The number of tips needs to be kept greater than 3
beta = 3

def main(arg=True):
    if os.path.exists('./dagSS/dagPool'):
        shutil.rmtree('./dagSS/dagPool')
    os.mkdir('./dagSS/dagPool')
    host_DAG = DAG(active_lst_addr='./dagSS/active_list.json',timespan=60)

# Generate the genesis block for DAG
    # genesisGen = os.popen(r"bash ./invokeRun.sh genesis")
    # genesisInfo = genesisGen.read()
    genesisInfo = 'QmaBYCmzPQ2emuXpVykLDHra7t8tPiU8reFMkbHpN1rRoo'
    print("The genesisBlock hash value is ", genesisInfo)
    # genesisGen.close()
    ini_trans = transaction.Transaction(time.time(), 0, 0.0, genesisInfo, [])
    transaction.save_genesis(ini_trans, './dagSS/dagPool/')
    ini_trans.name = 'GenesisBlock'
    ini_trans_file_addr = './dagSS/dagPool/'+ ini_trans.name +'.json'
    host_DAG.DAG_publish(ini_trans, beta)
    host_DAG.DAG_genesisDel()

    while arg:
        dagServer.socket_service("127.0.0.1", host_DAG, beta)

if __name__ == '__main__':

    main(True)