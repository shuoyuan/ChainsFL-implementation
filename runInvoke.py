# -*- coding: utf-8 -*-
# ******************************************************
# Filename     : runInvoke.py
# Author       : Shuo Yuan 
# Email        : ishawnyuan@gmail.com
# Blog         : https://iyuanshuo.com
# Last modified: 2020-06-04 17:43
# Description  : 
# ******************************************************

import os
import sys

if __name__ == '__main__':
    
    f = os.popen(r"./invokeParameters.sh" + " " + sys.argv[1] + " " + sys.argv[2])

    d = f.read()
    print(d)
    f.close()

