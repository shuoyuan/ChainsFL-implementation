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
        return outs, ipfsGet.poll()
    else:
        return errs, ipfsGet.poll()

def ipfsAddFile(fileName):
    ipfsAdd = subprocess.Popen(args=['ipfs add ' + fileName + ' | tr \' \' \'\\n\' | grep Qm'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    outs, errs = ipfsAdd.communicate(timeout=10)
    if ipfsAdd.poll() == 0:
        return outs, ipfsAdd.poll()
    else:
        return errs, ipfsAdd.poll()

if __name__ == '__main__':
    reCon, reCode = ipfsAddFile(sys.argv[1])
    print(reCon, '\n', reCode)