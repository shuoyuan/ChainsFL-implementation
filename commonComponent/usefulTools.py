# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : test.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-07-04 12:58
# Description  : 
# ******************************************************

import random
import sys
import json
import time
import subprocess

# Download the file using hashValue from ipfs network
def ipfsGetFile(hashValue, fileName):
    ipfsGet = subprocess.Popen(args=['ipfs get ' + hashValue + ' -o ' + fileName], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsGet.communicate(timeout=10)
    if ipfsGet.poll() == 0:
        return outs.strip(), ipfsGet.poll()
    else:
        return errs.strip(), ipfsGet.poll()

def ipfsAddFile(fileName):
    ipfsAdd = subprocess.Popen(args=['ipfs add ' + fileName + ' | tr \' \' \'\\n\' | grep Qm'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsAdd.communicate(timeout=10)
    if ipfsAdd.poll() == 0:
        return outs.strip(), ipfsAdd.poll()
    else:
        return errs.strip(), ipfsAdd.poll()

def queryLocal(lock, taskID, deviceID, currentEpoch, flagSet):
        localQuery = subprocess.Popen(args=['../commonComponent/interRun.sh query '+deviceID], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        outs, errs = localQuery.communicate(timeout=15)
        if localQuery.poll() == 0:
            localDetail = json.loads(outs.strip())
            if localDetail['epoch'] == currentEpoch and localDetail['taskID'] == taskID:
                print("The query result is ", outs.strip())
                while 1:
                    localFileName = './clientS/paras/' + taskID + deviceID + 'Epoch' + str(currentEpoch) + '.pkl'
                    outs, stt = ipfsGetFile(localDetail[], localFileName)
                    if stt == 0:
                        break
                    else:
                        print(outs.strip())
                lock.acquire()
                t1 = flagSet
                t1.add(deviceID)
                flagSet = t1
                lock.release()
        else:
            print("Failed to query this device!", errs)

if __name__ == '__main__':
    reCon, reCode = ipfsAddFile(sys.argv[1])
    print(reCon, '\n', reCode)