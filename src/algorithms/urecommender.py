# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:36:39 2017
测试不考虑跨平台特性下LSH的效果
令T = r = 10
@author: Yanchao
"""
import random
import numpy as np
from core.lsh_family import LSH
from algorithms.recommender import Recommender

class UCFRecommender(Recommender):
    def __init__(self, data, train_ratio = 0.99, ratio = 0.1, pool_size=10, hash_count=8):
        super().__init__(data, train_ratio, ratio)
        
        self.lsh_family = []
        self.hash_count = hash_count
        for i in range(self.hash_count):
            self.lsh_family.append(LSH(pool_size))
            self.lsh_family[i].fit(self.train_data.shape[1])
    
    def train(self):
        '''训练样本，即计算所有用户的hash值，
        然后根据其hash值放入不同的桶中'''
        self.buckets = []
        for i in range(self.hash_count):
            bucket = {}
            for j in range(self.train_rows):
                hash_value = self.lsh_family[i].get_hash_value(
                        self.train_samples[j])
                
                if bucket.get(hash_value) is None:
                    bucket[hash_value] = []
                
                bucket[hash_value].append(j)
            self.buckets.append(bucket)
            
    def find_similarity(self, x):
        '''查找指定样本的相似用户，以数组的形式把每个桶中的相似用户返回'''
        #计算样本x的hash值向量（每个hash函数都计算一次）
        similar_users = set()
        for i in range(self.hash_count):
            hash_value = self.lsh_family[i].get_hash_value(x)
            if self.buckets[i].get(hash_value) is not None:
                similar_users = similar_users.union(set(self.buckets[i][hash_value]))
        
        return list(similar_users)
    
    def predict(self):
        test_rows = len(self.test_samples)
        mae = 0
        count = 0
        
        for i in range(test_rows):
            predict_data = self.predict_response_time(self.test_samples[i])
            value = self.calculate_MAE(predict_data, self.remainders[self.train_rows + i])
            mae += value
            count += len(predict_data)
        
        result = 0
        if count > 0:
            result = mae/count
        else:
            print('count = 0')

        return result
    
    def predictNMae(self):
        test_rows = len(self.test_samples)
        mae = 0
        actualValue = 0
        count = 0
        
        for i in range(test_rows):
            predict_data = self.predict_response_time_org(self.test_samples[i])
            value, actual = self.calculate_NMAE(predict_data, self.remainders[self.train_rows + i])
            mae += value
            actualValue += actual
            count += len(predict_data)
        
            result = mae/actualValue

        return result

    def predict_response_time_org(self, x):
        similar_users = self.find_similarity(x)
        similar_num = len(similar_users)
        if similar_num == 0:
            return [0]
        
        predict_data = np.sum(self.remainders[similar_users], axis=0)
        predict_data = predict_data/similar_num
        return predict_data
        
    def predict_response_time(self, x):
        similar_users = self.find_similarity(x)

        predict_data = []
        similar_data = self.remainders[similar_users]
#        print(len(similar_users), '<---->', similar_data.shape)
        for i in range(similar_data.shape[1]):
            valid_data = [e for e in similar_data[:, i] if e > 0]
#            print('similar_data[:,i]=', similar_data[:, i], ', valid_data=', valid_data)
            if len(valid_data) == 0:
                predict_data.append(-1)
            else:
                predict_data.append(sum(valid_data)/len(valid_data))
        
        return predict_data
    
    def calculate_MAE(self, predict_data, actual_data):
        return abs(np.array(predict_data) - np.array(actual_data)).sum()
        
    def calculate_NMAE(self, predict_data, actual_data):
        return abs(np.array(predict_data) - np.array(actual_data)).sum(), np.array(actual_data).sum()


