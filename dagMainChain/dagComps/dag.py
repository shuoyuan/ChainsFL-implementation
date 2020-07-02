import os
import time
import json
# import transaction
import random


class DAG(object):
    def __init__(self,active_lst_addr='./dagSS/active_list.json',timespan =-1):
        """Inits the class."""
        self.active_lst_addr = active_lst_addr
        self.timespan = timespan
        self.active_pool = {}
        self.tips_pool = {} ##tips

    def DAG_add(self,trans):
        self.active_pool[trans.name] = trans.timestamp
        approve_lst = trans.apv_trans
        candidateDel = []
        if len(self.tips_pool) > 3:
            for ele in approve_lst:
                if ele in self.tips_pool:
                    del self.tips_pool[ele]
                    
## The number of tips needs to be kept greater than 3
## The DAG_publish function is responsible for saving the newly active_pool as json file and 
    def DAG_publish(self,trans,tipsMore):
        self.active_pool[trans.name] = trans.timestamp
        self.tips_pool[trans.name] = trans.timestamp
        approve_lst = trans.apv_trans
        if len(self.tips_pool) >= (len(approve_lst)+tipsMore):
            for ele in approve_lst:
                if ele in self.tips_pool:
                    del self.tips_pool[ele]
        elif (len(self.tips_pool) > tipsMore) and (len(self.tips_pool) < (len(approve_lst)+tipsMore)):
            tipsDelNum = len(self.tips_pool) - tipsMore
            while tipsDelNum != 0:
                tmpDelTip = approve_lst[random.randint(0,len(approve_lst)-1)]
                if tmpDelTip in self.tips_pool:
                    del self.tips_pool[tmpDelTip]
                    tipsDelNum -= 1
        with open(self.active_lst_addr,'w') as f:
            json.dump(self.active_pool,f)
            f.close()
        with open('./dagSS/tip_list.json','w') as f:
            json.dump(self.tips_pool,f)
            f.close()
    
    def DAG_genesisDel(self):
        if 'GenesisBlock' in self.tips_pool.keys():
            del self.tips_pool['GenesisBlock']
        with open('./dagSS/tip_list.json','w') as f:
            json.dump(self.tips_pool,f)
            f.close()

## !!! Note: This part is important for the realization of product logic.
    def DAG_choose(self,max_num):
        count = 0
        trans_lst = []
        for trans in self.tips_pool:
            if count >= max_num:
                break
            count+=1
            trans_lst.append(trans)
        return trans_lst