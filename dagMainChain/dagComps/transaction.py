import hashlib
import json
import numpy as np
import time

### transaction included three kinds of info: Authentication info, model parameters and approvement info
"""
timestamp: time
src_node: node that issued the transaction
model_acc: acc of the model
model_para: hash output from IPFS
apv_trans: list of rank
"""
class Transaction(object):
    def __init__(self,timestamp, src_node, model_acc, model_para ,apv_trans = []):
        self.timestamp = timestamp
        self.src_node = src_node
        self.model_acc = model_acc
        self.model_para = model_para
        self.apv_trans = apv_trans
        self.hashdata = self.hash()
        self.name = 'node{}_'.format(self.src_node) + str(self.timestamp)

    def hash(self):
        tohash = {
            'timestamp': self.timestamp,
            'src_node': self.src_node
        }

        str_data=json.dumps(
                            tohash,
                            default=lambda obj: obj.__dict__,
                            sort_keys=True)

        str_data_encode = str_data.encode()
        return hashlib.sha256(str_data_encode).hexdigest()

    def json_output(self):
        output = {
            'timestamp':self.timestamp,
            'src_node':self.src_node,
            'model_acc':self.model_acc,
            'model_para':self.model_para,
            'apv_trans':self.apv_trans
        }
        return output

    def __str__(self):
        return json.dumps(
            self.json_output(),
            default=lambda obj: obj.__dict__,
            sort_keys=True)

# transaction related functions
def read_transaction(trans_file):
    with open(trans_file, 'r') as f:
        trans_para = json.load(f)
        f.close()
    return Transaction(**trans_para)

def save_transaction(trans,file_addr):
    file_name = file_addr + '/node{}_'.format(trans.src_node) + str(trans.timestamp) + '.json'
    try:
        with open(file_name, 'w') as f:
            f.write(str(trans))
    except Exception as e:
        print("Couldn't save the trans " + file_name)
        print('Reason:', e)

def save_genesis(trans,file_addr):
    file_name = file_addr + 'GenesisBlock' + '.json'
    with open(file_name, 'w') as f:
        f.write(str(trans))
        f.close()

def name_2_time(trans_name):
    name = str(trans_name)
    split_lst = name.split('_',1)
    src_node = split_lst[0]
    time = float(split_lst[1])
    return time,src_node

if __name__ == '__main__':
    nodeNum = 1
    new_model_para = 'hash1'
    apv_trans_name = []
    new_trans = Transaction(time.time(), nodeNum,'', new_model_para, apv_trans_name)
    tst = new_trans.json_output()
    tst2 = json.dumps(tst).encode("utf-8")
    print(tst2)
    tst3 = json.loads(tst2.decode("utf-8"))
    print(tst3)
    print(tst3['timestamp'])
    