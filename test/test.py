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

def ipfsFileGet(hashValue, fileName):
    ipfsGet = subprocess.Popen(args=['ipfs get ' + hashValue + ' -o ' + fileName], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsGet.communicate(timeout=10)
    if ipfsGet.poll() == 0:
        return outs, ipfsGet.poll()
    else:
        return errs, ipfsGet.poll()
    

def shellResult():
    # shell envs
    shellEnv1 = "export PATH=${PWD}/../bin:$PATH"
    shellEnv2 = "export FABRIC_CFG_PATH=$PWD/../config/"
    shellEnv3 = "export CORE_PEER_TLS_ENABLED=true"
    shellEnv4 = "export CORE_PEER_LOCALMSPID=\"Org1MSP\""
    shellEnv5 = "export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
    shellEnv6 = "export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
    shellEnv7 = "export CORE_PEER_ADDRESS=localhost:7051"
    oneKeyEnv = shellEnv1 + " && " + shellEnv2 + " && " + shellEnv3 + " && " + shellEnv4 + " && " + shellEnv5 + " && " + shellEnv6 + " && " + shellEnv7

    # query task release info
    ## task info template {"Args":["set","taskRelease","{"taskID":"fl1234","epochs":10,"status":"start","usersFrac":0.1}"]} deviceID0000

    taskQueryshell = "peer chaincode query -C mychannel -n sacc -c '{\"Args\":[\"get\",\"taskRelease\"]}'"

    while 1:
        taskQuery = subprocess.Popen(args=[oneKeyEnv + " && " + taskQueryshell], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        try:
            outs, errs = taskQuery.communicate(timeout=15)
            print('The stdout is ', outs)
            print('The stderr is ', errs)
        except TimeoutExpired:
            taskQuery.kill()
        print('The returncode of taskQuery is ', taskQuery.poll())
        if taskQuery.poll() == 0:
            print('The query result is', outs)
            break
        else:
            print('The query failed to execute!\n')
            time.sleep(2)

def queryTest():
    taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
    taskEpochs = 10
    taskInitStatus = "start"
    taskUsersFrac = 0.1
    taskQuery = os.popen(r"./taskRelease.sh "+taskID+" "+str(taskEpochs)+" "+taskInitStatus+" "+str(taskUsersFrac))
    taskInfoR = taskQuery.read()
    print(taskInfoR)
    taskQuery.close()

def invokeTest():
    localFileName = 'device00010parameter.pkl'
    deviceName = ["device0002","device0003"]
    taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
    currentEpoch = 1
    taskQuery = os.popen(r"./invokeParameters.sh" + " " + localFileName + " " + deviceName[1]  + " " + taskID + " " + str(currentEpoch))
    taskInfoR = taskQuery.read()
    print(taskInfoR)
    taskQuery.close()

if __name__ == '__main__':
    a, resultCode = ipfsFileGet(sys.argv[1],sys.argv[2])
    print('a is ', a)
    print('resultCode is ', resultCode)