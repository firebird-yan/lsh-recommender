# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:37:50 2018
IPCC算法实现
@author: Yanchao
"""

import numpy as np
from algorithms.recommender import Recommender

class IPCCRecommender(Recommender):
    def __init__(self, data, train_ratio = 0.95, ratio = 0.05):
        super().__init__(data, train_ratio, ratio)
        self.num_of_services = self.data.shape[1]
        
    def predict(self):
        '''
        逐个预测所有用户所有未调用服务的响应时间，并计算其平均预测值
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
        top = 3
        indices = self.sample_indices[active_user]
        #计算active_user的平均值
        avg_value = np.average(self.data[active_user][indices == 1])
        
        similar_services, similarities = self.findSimilarServices(ws, top)
        
        predicted_value = 0
        totalSimilarities = 0
        
        for i in range(top):
            if indices[similar_services[i]] == 1:
                value = self.data[active_user][similar_services[i]]
                if value != -1:
                    predicted_value += similarities[i] * (value - avg_value)
                    totalSimilarities += similarities[i]
                    
        if totalSimilarities > 0:
            predicted_value /= totalSimilarities
            
        return predicted_value + avg_value

    def predictWhole(self):
        '''
        一次性预测所有测试用户的未调用的服务的响应时间
        这种方式的问题在于并不符合实际的应用需求
        '''
        sum_of_mae = 0.0
        num_of_test = len(self.test_data)
        
        for i in range(num_of_test):
            sum_of_mae += self.predictDataVector(i + self.num_of_train)
            
        return sum_of_mae/num_of_test
    
    def predictDataVector(self, active_user):
        '''
        预测用户的所有未调用的服务的响应时间
        这里直接采用矩阵运算，速度很快
        '''
        top = 3  #这里声明这个变量是为了将来方便修改
        similar_users, similarities = self.findSimilarUsers(active_user, top)
        indices = self.sample_indices[active_user]
        predicted_indices = (indices == 0) #这个变量经常计算，为了提高效率，先保存为变量
        avg_value = np.average(self.data[active_user][indices == 1])
        
        #置初始值
        expected_values = self.data[active_user][predicted_indices]
        predicted_values = np.zeros(expected_values.shape)
        
        total_similarities = 0.0
        for i in range(top):
            #计算相似用户可用服务的平均值
            indices = self.sample_indices[similar_users[i]]
            avg_value_of_similar = np.average(self.data[similar_users[i]][indices == 1]) 
            #将用户未调用过的（不管预测时候是否用得着）置为avg_value_of_similar，以便采用矩阵进行处理
            similar_user_data = self.data[similar_users[i]]
            similar_user_data[indices == 0] = avg_value_of_similar
            similar_user_data = similar_user_data[predicted_indices]
            #用矩阵的方式整体进行计算
            predicted_values += similarities[i] * (similar_user_data - avg_value_of_similar)
            total_similarities += similarities[i]
        
        predicted_values /= total_similarities
        predicted_values += avg_value
        
        return np.average(np.abs(predicted_values - expected_values))
        
    def findSimilarServices(self, index, top):
        ##这里如果计算速度很慢的话，可以考虑构建一个相似度矩阵
        similarities = []
        for i in range(self.num_of_services):
            if i != index:
                similarities.append(self.calculateSimilarity(index, i))
            
        return np.argsort(similarities)[0:top], np.sort(similarities)[0:top]

    def calculateSimilarity(self, test_index, train_index):
        #两个向量相乘，结果为1的元素即是二者的交集
        product = self.sample_indices[0:self.num_of_train,train_index] * \
            self.sample_indices[0:self.num_of_train,test_index]
        
        test = self.train_data[:,test_index][product == 1]
        train = self.train_data[:, train_index][product == 1]
        
        if len(test) == 0 or len(train) == 0:
            return -1
        
        avg_test = np.average(test)
        avg_train = np.average(train)
        
        numerator = np.dot(test - avg_test, train - avg_train)
        denominator = np.sqrt(np.dot(test - avg_test, test - avg_test)) * \
                    np.sqrt(np.dot(train - avg_train, train - avg_train))
        
        if denominator > 0:
            return numerator/denominator
        else:
            return -1

