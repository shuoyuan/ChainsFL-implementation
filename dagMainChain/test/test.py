# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : test.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-07-01 16:19
# Description  : 
# ******************************************************
import os

testtask = os.popen(r"bash ./invokeRun.sh genesis")
taskInfoR = testtask.read()
print("The genesisBlock hash value is ",taskInfoR)
testtask.close()