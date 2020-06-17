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
import os
import sys
import json

localFileName = 'device00010parameter.pkl'
deviceName = ["device0002","device0003"]
taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
currentEpoch = 1
taskQuery = os.popen(r"./invokeParameters.sh" + " " + localFileName + " " + deviceName[1]  + " " + taskID + " " + str(currentEpoch))
taskInfoR = taskQuery.read()
print(taskInfoR)
taskQuery.close()