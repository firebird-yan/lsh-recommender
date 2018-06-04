# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:03:29 2018
基类，定义了数据预处理的通用操作
@author: Yanchao
"""
import numpy as np
import random

class Recommender:
    def __init__(self, data, train_ratio, ratio):
        '''
        train_ratio:训练样本的比例, 剩余的为测试样本
        ratio: 抹去的训练样本中的数据的比例（即用户未使用的样本的比例）
        这里有两种方案，一种方案是训练样本和测试样本分别取样，即采用不同的ratio值；
        第二种方案是训练样本和测试样本的取值比例相同
        本方法采用第二种方案，先随机抽取数据，然后再区分训练样本和测试样本
        '''
        self.data = np.array(data)
        self.processed_data = np.copy(self.data)
        #直接随机取样本
        self.prepareData(ratio)
        self.splitData(train_ratio)
        
    def prepareData(self, ratio):
        '''
        从矩阵中随机抽取比例ratio的数据
        '''
        #声明一个和self.data同维度的矩阵来记录随机抽取的元素在矩阵中的下标
        #若被抽中则sample_indices矩阵对应的为1, 否则为0
        self.sample_indices = np.zeros(self.data.shape)
        #随机抽取比例为ratio的数据
        totalLength = self.data.size
        self.random_indices = random.sample(range(totalLength), 
                                            int(np.floor(totalLength * ratio)))
        self.sample_indices.ravel()[self.random_indices] = 1
        remain_indices = list(set(range(totalLength)) ^ set(self.random_indices))
        self.processed_data.ravel()[remain_indices] = 0
        
    def splitData(self, train_ratio):
        '''
        将数据分为训练样本（所占比例为train_ratio)和测试样本，
        '''
        self.num_of_train = int(np.floor(self.data.shape[0] * train_ratio))
        self.train_data = self.processed_data[0:self.num_of_train, :]
        self.test_data = self.processed_data[self.num_of_train:, :]
