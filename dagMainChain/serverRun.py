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
    if os.path.exists('./DAG/DAG_pool'):
        shutil.rmtree('./DAG/DAG_pool')
    os.mkdir('./DAG/DAG_pool')
    host_DAG = DAG(active_lst_addr='./DAG/active_list.json',timespan=60)

# Generate the genesis block for DAG
    # genesisGen = os.popen(r"bash ./invokeRun.sh genesis")
    # genesisInfo = genesisGen.read()
    genesisInfo = 'QmbFMke1KXqnYyBBWxB74N4c5SBnJMVAiMNRcGu6x1AwQH'
    print("The genesisBlock hash value is ", genesisInfo)
    # genesisGen.close()
    ini_trans = transaction.Transaction(time.time(), 0, 0.0, genesisInfo, [])
    transaction.save_transaction(ini_trans, './DAG/DAG_pool/')
    ini_trans.name = 'GenesisBlock'
    ini_trans_file_addr = './DAG/DAG_pool/'+ ini_trans.name +'.json'
    host_DAG.DAG_publish(ini_trans, beta)
    host_DAG.DAG_genesisDel()

    iteration_count = 0

    while arg:
        dagServer.socket_service("127.0.0.1")

if __name__ == '__main__':

    main(True)