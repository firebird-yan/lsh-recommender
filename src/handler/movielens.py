
'''
本类专门用于处理MovieLens的数据，包括读取文件，返回相应的数据等
'''

import numpy as np

class MovieLensHandler:
    def __init__(self):
        self.dir = '../../datasets/MovieLens/'
        self.num_of_users = 6040
        self.num_of_movies = 3952
        self.data = np.zeros((self.num_of_users, self.num_of_movies))

        pass

    def readRatings(self):
        filename = self.dir + '/ratings.dat'

        with open(filename, "r") as f:
            for line in f:
                vals = line.split('::')
                user_index = int(vals[0]) - 1
                movie_index = int(vals[1]) - 1
                self.data[user_index][movie_index] = int(vals[2])

        return self.data

    def findIntersections(self):
        '''
        本方法针对每一个用户，查找和该用户的评分电影交集数小于等于1的所有用户，
        将结果保存到一个二维数组里，并返回
        '''
        # 这里如果是乘以1的话，即令ratings矩阵的类型为int类型，则运算速度非常慢，大约120s，
        # 但替换为乘以1.0后，只需2s。
        ratings = (self.data > 0) * 1.0
        intersections = np.dot(ratings, ratings.T)
        return intersections


#
# handler = MovieLensHandler()
# handler.readRatings()
# print('loaded!')
# intersections = handler.findIntersections()
# print('computed!')
# for i in range(10):
#     for j in range(100):
#         if intersections[i][j] <= 1:
#             print(i, ',', j)
#
# print('finished!')