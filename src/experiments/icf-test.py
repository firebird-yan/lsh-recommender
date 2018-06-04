# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:42:15 2018
测试实验结果
@author: Yanchao
"""
from algorithms.urecommender import UCFRecommender
from algorithms.irecommender import ICFRecommender
from algorithms.irecommender3 import ICFRecommenderWithMean
from algorithms.upcc import UPCCRecommender
from algorithms.ipcc import IPCCRecommender
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt

class Test:
    def __init__(self, r_ratio, c_ratio, pool_size, hash_count):
        self.data = np.array(self.read_rt_matrix())
        self.rows = self.data.shape[0]
        self.columns = self.data.shape[1]
        
        self.r_ratio = r_ratio
        self.c_ratio = c_ratio
        self.pool_size = pool_size
        self.hash_count = hash_count
        self.times = 100
    
    def read_rt_matrix(self):
        rt_data = []
        with open('../../datasets/ws-dream/tpMatrix.txt', 'r') as f:
            for line in f:
                values = [float(x) for x in line.split()]
                rt_data.append(values)
                
        return rt_data
    
    def sample(self, r, c):
        '''随机选取row个用户，column个服务'''
        row_indices = random.sample(range(self.rows), r)
        column_indices = random.sample(range(self.columns), c)

        sample_data = self.data[:, column_indices]
        sample_data = sample_data[row_indices, :]
        
        return sample_data

    def runUCF(self, sample_data, ratio, pool_size, hash_count):
        recommender = UCFRecommender(sample_data, ratio,  
                                     pool_size, hash_count)
        
        recommender.train()
        
        mae = recommender.predict()
        
        return mae 
    
    def runICF(self, sample_data, r_ratio, c_ratio, pool_size, hash_count):
#        anchor = time.time()
        recommender = ICFRecommender(sample_data, r_ratio, c_ratio, 
                                     pool_size, hash_count)
#        print('sample cost:', time.time()- anchor, end=',')
        
#        anchor = time.time()
        recommender.train()
#        print('train cost:', time.time()- anchor, end=',')
        
        anchor = time.time()
        # 目前实际上返回的是NMAME
        mae = recommender.predict()
        
        return mae, time.time() - anchor
    
    def runICF2(self, sample_data, r_ratio, c_ratio, pool_size, hash_count):
#        anchor = time.time()
        recommender = ICFRecommenderWithMean(sample_data, r_ratio, c_ratio, 
                                     pool_size, hash_count)
#        print('sample cost:', time.time()- anchor, end=',')
        
#        anchor = time.time()
        recommender.train()
#        print('train cost:', time.time()- anchor, end=',')
        
        anchor = time.time()
        # 目前实际上返回的是NMAME
        mae = recommender.predict()
        
        return mae, time.time() - anchor
    
    def runUPCC(self, sample_data, train_ratio, ratio):
        recommender = UPCCRecommender(sample_data, train_ratio, ratio)
        
        anchor = time.time()
        mae = recommender.predict()
        
        return mae, time.time() - anchor
    
    def runIPCC(self, sample_data, train_ratio, ratio):
        recommender = IPCCRecommender(sample_data, train_ratio, ratio)
        
        anchor = time.time()
        mae = recommender.predict()
        
        return mae, time.time() - anchor

    
    def constrastICFWithUPCC(self, num_of_users, num_of_services):
        icf_maes = []
        upcc_maes = []
        
        icf_costs = []
        upcc_costs = []
        for i in range(self.times):
            sample_data = self.sample(num_of_users, num_of_services)
            
            mae,cost = self.runICF(sample_data, self.r_ratio, self.c_ratio, self.pool_size, self.hash_count)
            icf_maes.append(mae)
            icf_costs.append(cost)
            mae,cost = self.runUPCC(sample_data, self.r_ratio, self.c_ratio)
            upcc_maes.append(mae)
            upcc_costs.append(cost)
            
            print('>', end='')
            if (i + 1) % 20 == 0:
                print('')
        
        print(np.average(icf_maes), '\t', np.average(upcc_maes))
        
        
    def contrastICFWithUPCCandIPCC(self, num_of_users, num_of_services):
        icf_maes = []
        upcc_maes = []
        ipcc_maes = []
        
        icf_costs = []
        upcc_costs = []
        ipcc_costs = []
        for i in range(self.times):
            sample_data = self.sample(num_of_users, num_of_services)
            
            mae,cost = self.runICF(sample_data, self.r_ratio, self.c_ratio, self.pool_size, self.hash_count)
            icf_maes.append(mae)
            icf_costs.append(cost)
            mae,cost = self.runUPCC(sample_data, self.r_ratio, self.c_ratio)
            upcc_maes.append(mae)
            upcc_costs.append(cost)
            mae,cost = self.runIPCC(sample_data, self.r_ratio, self.c_ratio)
            ipcc_maes.append(mae)
            ipcc_costs.append(cost)
            
            print('>', end='')
            if (i + 1) % 20 == 0:
                print('')
        
        data = {'icf_maes':icf_maes, 'upcc_maes':upcc_maes, 'ipcc_maes':ipcc_maes, 
                'icf_costs':icf_costs, 'upcc_costs':upcc_costs, 'ipcc_costs':ipcc_costs}
        
        filename = '../../outputs/icf/contrast-%d-%d.json'%(num_of_users, num_of_services)
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        print('===================', filename, 'created!==========')
    
    def icfVariedWithHashCountAndPoolSize(self):
        hash_count_options = [6, 8, 10, 12, 14]
        pool_size_options = [6, 8, 10, 12, 14]
        
        hash_count_num = len(hash_count_options)
        pool_size_num = len(pool_size_options)
        
        maes = np.zeros((hash_count_num, pool_size_num))
        costs = np.zeros(maes.shape)
        for t in range(self.times):
            sample_data = self.sample(300, 300)
            for i in range(hash_count_num):
                for j in range(pool_size_num):
                    mae, cost = self.runICF(sample_data, self.r_ratio,
                                                 self.c_ratio, 
                                                 pool_size_options[j],
                                                 hash_count_options[i])
                    maes[i][j] += mae
                    costs[i][j] += cost
                    print('>', end='')
            print('')
        maes /= self.times
        costs /= self.times

        data = {'maes':maes.tolist(), 'costs':costs.tolist()}
        with open('../../outputs/icf/icf-lsh-300-300-2.json', 'w') as f:
            json.dump(data, f)
            
    def icfTest(self):
        icf_maes = []
        icf2_maes = []
        for i in range(self.times):
            sample_data = self.sample(300, 300)
            
            mae = self.runICF(sample_data, self.r_ratio, self.c_ratio, self.pool_size, self.hash_count)
            icf_maes.append(mae)
            mae = self.runICF2(sample_data, self.r_ratio, self.c_ratio, self.pool_size, self.hash_count)
            icf2_maes.append(mae)
            print(i + 1)
        
        plt.plot(range(1, self.times + 1), icf_maes, 'y-')
        plt.plot(range(1, self.times + 1), icf2_maes, 'r-')
        print("icf:", np.average(icf_maes), "icf_with_mean:", np.average(icf2_maes))
            
     
    def adjustLSHParameters(self):
        '''查看一下hash_count和pool_size对mae的值影响，最好找到稳定值'''
        options = [6, 8, 10, 12, 14, 16]
        opt_num = len(options)
        maes_array = []
        
        for i in range(self.times):
            sample_data = self.sample(300, 1000)
            maes = np.zeros((opt_num, opt_num))
            for m in range(opt_num): #index of hash_count
                for n in range(opt_num): #index of pool_size
                    maes[m][n] = self.runICF(sample_data, self.r_ratio, self.c_ratio,
                                      options[n], options[m])
                    
            maes_array.append(maes)
            
        with open('../../outputs/icf/maes_300_1000_100.json', 'w', encoding='utf-8') as wf:
            json.dump(maes_array, wf)
    
        
        
test = Test(0.99, 0.1, 10, 10)
#test.adjustLSHParameters()        
#test.upccTest()
num_of_users = [50]#, 100, 150, 200, 250, 300]
num_of_services = [100]#, 200, 300, 400]
for u in num_of_users:
    for s in num_of_services:        
        test.contrastICFWithUPCCandIPCC(u, s)
#test.icfVariedWithHashCountAndPoolSize()

#test.constrastICFWithUPCC(300, 600)
#test.constrastICFWithUPCC(300, 500)