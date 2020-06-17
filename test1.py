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

taskID = 'task'+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))+str(random.randint(1,10))
taskEpochs = 10
taskInitStatus = "start"
taskUsersFrac = 0.1
taskQuery = os.popen(r"./taskRelease.sh "+taskID+" "+str(taskEpochs)+" "+taskInitStatus+" "+str(taskUsersFrac))
taskInfoR = taskQuery.read()
print(taskInfoR)
taskQuery.close()
