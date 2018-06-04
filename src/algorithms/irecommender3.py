# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:51:46 2018
Item-based CF using LSH
@author: Yanchao
"""

import numpy as np
from algorithms.recommender import Recommender
from core.lsh_family import LSH

        
class ICFRecommenderWithMean(Recommender):
        
    def __init__(self, data, train_ratio = 0.95, ratio = 0.1, 
                 pool_size=10, hash_count=8):
        super().__init__(data, train_ratio, ratio)
        self.num_of_services = self.data.shape[1]
        self.lsh_family = []
        self.hash_count = hash_count
        for i in range(self.hash_count):
            self.lsh_family.append(LSH(pool_size))
            ##这里拟合的是和UCFRecommender的标准不一样
            self.lsh_family[i].fit(self.num_of_train)
    
    def train(self):
        '''训练样本，即计算所有服务的hash值，
        然后根据其hash值放入不同的桶中'''
        self.buckets = []
        for i in range(self.hash_count):
            bucket = {}
            for s in range(self.num_of_services):
                hash_value = self.lsh_family[i].get_hash_value(
                        self.train_data[:, s])
                
                if bucket.get(hash_value) is None:
                    bucket[hash_value] = []
                
                bucket[hash_value].append(i)
            self.buckets.append(bucket)
            
    def find_similar_services(self, y):
        '''查找指定样本的相似用户，以数组的形式把每个桶中的相似用户返回'''
        #计算样本x的hash值向量（每个hash函数都计算一次）
        similar_services = set()
        for i in range(self.hash_count):
            hash_value = self.lsh_family[i].get_hash_value(y)
            ##这里要注意的是，如果用户数过少，那么hash_value相同的值越多
            ##速度会变慢
            if self.buckets[i].get(hash_value) is not None:
                for s in self.buckets[i][hash_value]:
                    similar_services.add(s)
        
        return list(similar_services)
    
    def predict(self):
        '''
        逐个预测所有用户所有未调用服务的响应时间，计算并返回mae
        '''
        sum_of_mae = 0.0
        num_of_test = len(self.test_data)
        total_predict = 0
        
        columns = self.data.shape[1]
        for i in range(num_of_test):
            index = i + self.num_of_train
            indices = self.sample_indices[index]
            for ws in range(columns):
                if indices[ws] == 0:
                    sum_of_mae += self.predictData(i + self.num_of_train, ws)
                    total_predict += 1
            
        return sum_of_mae/total_predict
    
    
    def predictData(self, active_user, ws):
        '''
        预测用户active_user调用ws的时间，
        并返回其预测值
        '''
        indices = self.sample_indices[active_user]
        #计算active_user的平均值
        avg_value = np.average(self.data[active_user][indices == 1])
        
        similar_services = self.find_similar_services(self.train_data[:, ws])
        
        predicted_value = 0
        total = 0
        
        for i in similar_services:
            if indices[i] == 1:
                value = self.data[active_user][i]
                if value != -1:
                    predicted_value += (value - avg_value)
                    total += 1
                    
        if total > 0:
            predicted_value /= total
            
        return predicted_value + avg_value
 
    def predictWhole(self):
        test_num = len(self.test_data)        
        sum_of_mae = 0.0
        
        for i in range(test_num):
            mae = self.predictData(self.num_of_train + i)
            sum_of_mae += mae
        
        return sum_of_mae/test_num

    def predictWholeData(self, active_user):
        '''
        actuve_user:测试样例在self.data的行坐标
        预测样本active_user从未调用的服务的响应时间，
        计算并返回其MAE值
        '''
        indices = self.sample_indices[active_user]
        
        unavailable_indices = (indices == 0)
        predicted_indices = indices[unavailable_indices]
        
        expected_value = self.data[active_user][unavailable_indices]
        predicted_value = np.zeros(expected_value.shape)
        
        x = self.data[active_user]
        num_of_predicted = len(predicted_indices)
        for i in range(num_of_predicted):
            current_indices = np.zeros(x.shape)
            similar_services = self.find_similar_services(self.train_data[:, i])
            current_indices[similar_services] = 1
            current_indices[unavailable_indices] = 0
            
            available_data = x[current_indices == 1]
            if len(available_data) > 0:
                predicted_value[i] = np.average(available_data)
                
        return np.average(np.abs(expected_value - predicted_value))
            
 
