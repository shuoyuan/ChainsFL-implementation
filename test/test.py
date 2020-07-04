# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : test.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-14 20:01
# Description  : 
# ******************************************************

import random
import sys
import json
import time
import subprocess
import threading
# import func_timeout
# import time

# from func_timeout import func_set_timeout

def taskRelease():
    iteration_count = 0
    taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
    if iteration_count == 0:
        taskEpochs = 10
        taskInitStatus = "start"
        taskUsersFrac = 0.1
        taskRelease = subprocess.Popen(args=['../interRun.sh release '+taskID+' '+str(taskEpochs)+' '+taskInitStatus+' '+str(taskUsersFrac)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        trOuts, trErrs = taskRelease.communicate(timeout=10)
        if taskRelease.poll() == 0:
            print(trOuts.strip())
            print('*** ' + taskID + ' has been released! ***\n')
        else:
            print(trErrs.strip())
            print('*** Failed to release ' + taskID + ' ! ***\n')

def ipfsFileGet(hashValue, fileName):
    ipfsGet = subprocess.Popen(args=['ipfs get ' + hashValue + ' -o ' + fileName], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsGet.communicate(timeout=10)
    if ipfsGet.poll() == 0:
        return outs, ipfsGet.poll()
    else:
        return errs, ipfsGet.poll()    

def shellResult(lock):
    # shell envs
    shellEnv1 = "export PATH=/home/shawn/Documents/fabric-samples/test-network/../bin:$PATH"
    shellEnv2 = "export FABRIC_CFG_PATH=/home/shawn/Documents/fabric-samples/test-network/../config/"
    shellEnv3 = "export CORE_PEER_TLS_ENABLED=true"
    shellEnv4 = "export CORE_PEER_LOCALMSPID=\"Org1MSP\""
    shellEnv5 = "export CORE_PEER_TLS_ROOTCERT_FILE=/home/shawn/Documents/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
    shellEnv6 = "export CORE_PEER_MSPCONFIGPATH=/home/shawn/Documents/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
    shellEnv7 = "export CORE_PEER_ADDRESS=localhost:7051"
    oneKeyEnv = shellEnv1 + " && " + shellEnv2 + " && " + shellEnv3 + " && " + shellEnv4 + " && " + shellEnv5 + " && " + shellEnv6 + " && " + shellEnv7

    # query task release info
    ## task info template {"Args":["set","taskRelease","{"taskID":"fl1234","epochs":10,"status":"start","usersFrac":0.1}"]} deviceID0000

    taskQueryshell = "peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"deviceID0000\"]}'"

    while 1:
        taskQuery = subprocess.Popen(args=[oneKeyEnv + " && " + taskQueryshell], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        outs, errs = taskQuery.communicate(timeout=15)
        if taskQuery.poll() == 0:
            print('Success to execute this command!', outs.strip())
            # test.append(outs.strip())
            global test
            lock.acquire()
            t1 = test
            t1.append(outs.strip())
            test = t1
            lock.release()
            break
        else:
            print("Failed to execute this command!", errs)
            return errs, taskQuery.poll()

def invokeTest():
    localFileName = 'device00010parameter.pkl'
    deviceName = ["device0002","device0003"]
    taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
    currentEpoch = 1
    taskQuery = os.popen(r"./invokeParameters.sh" + " " + localFileName + " " + deviceName[1]  + " " + taskID + " " + str(currentEpoch))
    taskInfoR = taskQuery.read()
    print(taskInfoR)
    taskQuery.close()

def invokeTest(deviceID):
    while 1:
        fileHashtst = 'file' + str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
        taskQuery = subprocess.Popen(args=['../interRun.sh local '+deviceID+' task1234 1 '+fileHashtst], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        outs, errs = taskQuery.communicate(timeout=15)
        if taskQuery.poll() == 0:
            print('Success to execute this command!', outs.strip())
            break
        else:
            print("Failed to execute this command!", errs)
            return errs, taskQuery.poll()

def queryTest(lock,deviceID):
    while 1:
        taskQuery = subprocess.Popen(args=['../interRun.sh query '+deviceID], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        outs, errs = taskQuery.communicate(timeout=15)
        if taskQuery.poll() == 0:
            print('Success to execute this command!', outs.strip())
            # test.append(outs.strip())
            global test
            lock.acquire()
            t1 = test
            t1.add(deviceID)
            test = t1
            lock.release()
            break
        else:
            print("Failed to execute this command!", errs)
            return errs, taskQuery.poll()

def queryLocal(lock, taskID, deviceID, currentEpoch, test):
    localQuery = subprocess.Popen(args=['../interRun.sh query '+deviceID], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = localQuery.communicate(timeout=15)
    if localQuery.poll() == 0:
        localDetail = json.loads(outs.strip())
        if localDetail['epoch'] == currentEpoch and localDetail['taskID'] == taskID:
            print(outs.strip())
            lock.acquire()
            t1 = test
            t1.add(localDetail['paras'])
            test = t1
            lock.release()
    else:
        print("Failed to query this device!", errs)

if __name__ == '__main__':

    allDeviceName = []
    currentEpoch = 1
    taskID = 'task1234'
    for i in range(100):
        allDeviceName.append("device"+("{:0>5d}".format(i)))
    start = time.time()
    test = set()
    ts = []
    lock = threading.Lock()
    for deviceID in allDeviceName:
        t = threading.Thread(target=queryLocal,args=(lock,taskID,deviceID,currentEpoch,test,))
        t.start()
        ts.append(t)
    for t in ts:
        t.join()

    elapsed = (time.time() - start)
    print(elapsed)
    # time.sleep(5)
    # print(allDeviceName)
    print(len(test))
    print(test)



    # # Batch invoke
    # allDeviceName = []
    # for i in range(100):
    #     allDeviceName.append("device"+("{:0>5d}".format(i)))
    # start = time.time()

    # for deviceID in allDeviceName:
    #     t = threading.Thread(target=invokeTest,args=(deviceID,))
    #     t.start()
    # elapsed = (time.time() - start)
    # print(elapsed)
    # print(allDeviceName)