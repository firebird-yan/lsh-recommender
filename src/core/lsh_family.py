# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:56:49 2017
LSH函数族实现，函数族中hash value的位数由pool_size决定
使用前需先调用fit函数为每个hash位生成随机向量paramerters
每个hash位就是用parameters[i]与数据作点乘，如果>0,hash值为1，否则为0
@author: Yanchao
"""
import numpy as np
import random
class LSH:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.parameters = None
        
    def fit(self, dimensions):
        self.parameters = np.zeros((dimensions, self.pool_size))
        self.weights = np.zeros((self.pool_size, 1)) #计算hash值的权重向量[2^0, 2^1, 2^2, ...]'
        self.dimensions = dimensions

        weight = 1
        for i in range(self.pool_size):
            self.weights[i][0] = weight
            weight *= 2

            for j in range(self.dimensions):
                self.parameters[j][i] = random.uniform(-1, 1)

    def get_hash_value(self, x):
        '''
        计算x的hash值
        得到一个1*pool_size维的向量，将向量中值大于0的置为1，小于等于0的置0
        然后将该向量转换为十进制，如[1 0 1 0]转换为5
        :param x:
        :return:
        '''
        # x = x.reshape(1, self.dimensions)
        values = np.dot(x, self.parameters)
        values = (values > 0).astype(float)
        return np.squeeze(np.dot(values, self.weights))

    def get_all_hash_value(self, X):
        '''
        以矩阵运算的方式同时计算所有向量的hash值
        :param X:
        :return:
        '''
        X = X.reshape(1, self.dimensions)
        # print(X.shape, self.parameters.shape, self.weights.shape)
        values = np.dot(X, self.parameters)
        values = (values > 0).astype(float)
        return np.squeeze(np.dot(values, self.weights))
        
        

