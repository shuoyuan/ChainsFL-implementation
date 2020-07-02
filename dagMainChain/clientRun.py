import sys
import clientutils
import transaction
import socket
import os
from dagSocket import dagClient
from dagComps.dag import DAG as DAG
from federatedLearning
import time
import threading
import shutil
import json

# The number of tips confirmed by the new transaction
alpha = 2
## The number of tips needs to be kept greater than 3
beta = 3

nodeNum = 1

def main(aim_addr='127.0.0.1',arg3=2,arg4=3,arg5=100,arg6=5,arg7=1e-4):
    if os.path.exists('./DAG/DAG_pool'):
        shutil.rmtree('./DAG/DAG_pool')
    os.mkdir('./DAG/DAG_pool')

    with open('./run_state.txt','w') as f:
        f.write('-1')
        f.close()

    print('Detect status of the task')
    while 1:
        with open('./run_state.txt', 'r') as f:
            state = f.read()
            if state != '-1':
                f.close()
                break
        time.sleep(1)
        print('Task not started!')

    print('Task is runing!')

    with open('./run_state.txt', 'r') as f:
        rank = int(f.read())
    print('The epoch of this sharding is ' + str(rank))

    iteration_count = 0

    while 1:
        iteration_count += 1
        apv_trans_name = []
        if iteration_count == 1:
            apv_trans_name.append('GenesisBlock')
        else:
            tips_list = 'tip_list'
            tips_file = './DAG/DAG_pool/' + tips_list + '.json'
            dagClient.client_tips_require(aim_addr, tips_list, tips_file)
            with open(self.tips_file,'r') as f1:
                tips_dict = json.load(f1)
                f1.close()
            if len(tips_dict) <= alpha:
                apv_trans_name = list(tips_dict.keys())
            else:
                apv_trans_name = random.sample(tips_dict.keys(), alpha)
        for apvTrans in apv_trans_name:
            apvTransFile =  './DAG/DAG_pool/' + apvTrans + '.json'
            dagClient.client_trans_require(aim_addr, apvTrans, apvTransFile)
        
        if len(trans_name_lst) != 0:
            trans_lst = []
            for name in trans_name_lst:
                trans_lst.append(transaction.read_transaction('./DAG/DAG_pool/'+name+'.json'))
            client_CNN.clear()
            while 1:
                try:
                    apv_trans = client_CNN.load_validate_approve(trans_lst,'./data',arg3)
                    client_CNN.compute_global()
                    client_CNN.train('./data')
                except Exception:
                    print("Error in clientrun.train")
                    client_CNN.clear()
                else:
                    break
            new_model_para = client_CNN.back_model()
            apv_trans_name = []
            for ele in apv_trans:
                apv_trans_name.append(ele.name)
            new_trans = transaction.Transaction(time.time(), nodeNum,'', new_model_para, apv_trans_name)
            transaction.save_transaction(new_trans, './DAG/DAG_pool/')
            # client_DAG.DAG_publish(new_trans)
        time.sleep(10)

if __name__ == '__main__':
    main(2,'192.168.0.104',2,3,1000,5,1e-1)