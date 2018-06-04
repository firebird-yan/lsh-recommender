
'''
本文件基于“敌人的敌人是朋友”这一社会化理论，为了解决冷启动问题，
实现了一种通过查找敌人（完全不相似的用户），间接找到相似用户的方法
'''

from handler.movielens import MovieLensHandler
from core.lsh_family import LSH
import numpy as np
import random
import time


class EnemyRecommender:
    '''
    processed_data: 经过稀疏化处理后的数据
    train_data: 过滤掉与user用户相似的用户后的数据
    '''
    def __init__(self, data, ratio = 0.8,
                 pool_size = 10, hash_count = 8):
        '''
        :param data: 从文件中读取的评分数据，矩阵形式，行数为用户数，列数为项目数
        :param user：测试用户的索引
        :param ratio: 为了让测试预测的效果，设定保留的数据的比例，剩余的数据抹零
        :param pool_size: 每个LSH返回的hash值的位数（二进制）
        :param hash_count: LSH的个数
        '''
        # self.num_of_users = self.data.shape[0]
        self.data = data
        self.processed_data = np.copy(self.data)

        self.sample(ratio)

        self.user = np.random.randint(0, self.data.shape[0])
        self.remove_similar_users()

        self.lsh_family = []
        self.hash_count = hash_count
        for i in range(self.hash_count):
            self.lsh_family.append(LSH(pool_size))
            #这里是查找用户的相似度
            self.lsh_family[i].fit(self.data.shape[1])

    def remove_similar_users(self, threshold=0.1):
        filter = []
        num_of_data = self.data.shape[0]
        u = self.data[self.user]
        for j in range(num_of_data):
            if self.compute_similarity_between_users(u, self.data[j]) < threshold:
                filter.append(j)

        self.train_data = self.processed_data[filter]

    def compute_similarity_between_users(self, u, v):
        intersection = (np.multiply(u, v) > 0)

        u = u[intersection]
        v = v[intersection]

        if len(u) == 0:
            return 0

        u_bias = u - np.average(u)
        v_bias = v - np.average(v)

        numerator = np.dot(u_bias, v_bias.T)
        denominator = np.sqrt(np.dot(u_bias, u_bias.T) * np.dot(v_bias, v_bias.T))

        if denominator == 0:
            return -1

        return numerator / denominator

    def sample(self, ratio):
        '''
        从矩阵中随机抽取比例ratio的数据
        '''
        # 声明一个和self.data同维度的矩阵来记录随机抽取的元素在矩阵中的下标
        # 若被抽中则sample_indices矩阵对应的为1, 否则为0
        self.sample_indices = np.zeros(self.data.shape)
        # 随机抽取比例为ratio的数据
        totalLength = self.data.size
        self.random_indices = random.sample(range(totalLength),
                                            int(np.floor(totalLength * ratio)))
        self.sample_indices.ravel()[self.random_indices] = 1
        remain_indices = list(set(range(totalLength)) ^ set(self.random_indices))
        self.processed_data.ravel()[remain_indices] = 0

    def compute_lsh(self):
        '''
        计算每个用户的lsh值，每个用户的lsh值计算hash_count次
        :return:
        '''
        self.buckets = []
        num_of_users = self.train_data.shape[0]

        for i in range(self.hash_count):
            bucket = {}
            hash_values = self.lsh_family[i].get_hash_value(self.train_data)

            for u in range(num_of_users):
                hash_value = hash_values[u]
                if bucket.get(hash_value) is None:
                    bucket[hash_value] = []

                bucket[hash_value].append(u)

            self.buckets.append(bucket)

    def find_similar_users(self, u):
        similar_users = set()

        for i in range(self.hash_count):
            hash_value = np.asscalar(self.lsh_family[i].get_hash_value(u))

            if self.buckets[i].get(hash_value) is not None:
                for user in self.buckets[i][hash_value]:
                    similar_users.add(user)

        return list(similar_users)

    def find_enemies(self, u):
        '''
        查找用户的"敌人"，即hash值完全不同（每位互异）
        加入pool_size=8，即hash_value对应的二进制位为8位，
        则对应的敌人的hash_value为255-get_hash_value(8)
        为了验证实验，我们需要将用户u的相似度比较高的用户从训练集中删掉
        :param u:
        :return:
        '''
        enemies = set()
        mask = np.power(2, self.lsh_family[0].pool_size) - 1

        for i in range(self.hash_count):
            hash_value = np.asscalar(self.lsh_family[i].get_hash_value(u))

            enemy_hash_value = mask - hash_value

            if self.buckets[i].get(enemy_hash_value) is not None:
                for user in self.buckets[i][enemy_hash_value]:
                    enemies.add(user)

        return list(enemies)

    def find_enemies_of_enemies(self, u):
        '''
        查找u的敌人的敌人
        :param u: users u
        :return:
        '''
        enemies_of_enemies = set()

        enemies = self.find_enemies(u)
        for e in enemies:
            #这里enemies都是用户在训练集中的索引值
            enemies_of_enemies = enemies_of_enemies.union(self.find_enemies(self.train_data[e]))

        return enemies_of_enemies

    def similarities_test(self, u):
        intersection = (np.multiply(u, self.train_data) > 0) * 1.0
        u_bias = np.multiply(intersection, u.T)
        count = np.sum(intersection, axis=1)
        count[count == 0] = 1 #该语句用于处理交集为空的情况
        u_average = np.sum(u_bias, axis = 1)/count
        u_average = u_average.reshape(self.train_data.shape[0], 1)
        u_bias = np.multiply(u_bias - u_average, intersection)

        train_bias = np.multiply(intersection, self.train_data)
        train_average = np.sum(train_bias, axis = 1)/count
        train_average = train_average.reshape(self.train_data.shape[0], 1)
        train_bias = np.multiply(train_bias - train_average, intersection)

        u_norm = np.linalg.norm(u_bias, axis=1)
        u_norm = u_norm.reshape(self.train_data.shape[0], 1)
        u_norm[u_norm == 0] = 1
        train_norm = np.linalg.norm(train_bias, axis=1)
        train_norm = train_norm.reshape(self.train_data.shape[0], 1)
        train_norm[train_norm == 0] = 1

        train_bias = train_bias/train_norm
        u_bias = u_bias/u_norm

        # similarites = np.diag(np.dot(u_bias, train_bias.T))
        similarites = np.zeros((self.train_data.shape[0], 1))
        for i in range(self.train_data.shape[0]):
            similarites[i] = np.dot(train_bias[i], u_bias[i])

        return similarites

    def similarities_test2(self, u):
        similarites = np.zeros(self.train_data.shape[0])
        for i in range(self.train_data.shape[0]):
            similarites[i] = self.compute_similarity_between_users(u, self.train_data[i])

        return similarites

    def find_enemies_by_pearson(self, u):
        similarites = np.zeros(self.train_data.shape[0])
        for i in range(self.train_data.shape[0]):
            similarites[i] = self.compute_similarity_between_users(u, self.train_data[i])

        indices = np.array(range(self.train_data.shape[0]))
        # print(len(indices), ',', len(indices[similarites == -1]))
        return indices[similarites == -1]

    def predict_with_pearson(self):
        similar_users = set()
        enemies = self.find_enemies_by_pearson(self.processed_data[self.user])
        for e in enemies:
            similar_users = similar_users.union(self.find_enemies_by_pearson(self.train_data[e]))

        similar_users = list(similar_users)

        similar_users_data = self.train_data[similar_users]
        mae = 0
        index = self.train_data.shape[0]
        total = 0

        for j in range(self.data.shape[1]):
            if self.sample_indices[self.user][j] == 0 and self.data[self.user][j] > 0:
                columns = similar_users_data[:, j]
                # print('before:', len(columns), end=',')
                columns = columns[columns > 0]
                # print('after:', len(columns))

                if len(columns) > 0:
                    value = np.average(columns)
                else:
                    v = self.train_data[:, j]
                    v = v[v > 0]
                    if len(v) > 0:
                        value = np.average(v)
                    else:
                        value = 0

                mae += np.abs(value - self.data[self.user][j])
                total += 1

        return mae / total

    def predict(self):
        '''
        预测用户u的被屏蔽的值，返回其MAE
        :param u:
        :return:
        '''
        similar_users = list(self.find_enemies_of_enemies(self.processed_data[self.user]))
        similar_users_data = self.train_data[similar_users]
        mae = 0
        index = self.train_data.shape[0]
        total = 0

        for j in range(self.data.shape[1]):
            if self.sample_indices[self.user][j] == 0 and self.data[self.user][j] > 0:
                columns = similar_users_data[:, j]
                # print('before:', len(columns), end=',')
                columns = columns[columns > 0]
                # print('after:', len(columns))

                if len(columns) > 0:
                    value = np.average(columns)
                else:
                    v = self.train_data[:, j]
                    v = v[v > 0]
                    if len(v) > 0:
                        value = np.average(v)
                    else:
                        value = 0

                mae += np.abs(value - self.data[self.user][j])
                total += 1

        return mae/total

    def predict_with_average(self):
        '''
        采用求平均值的方式预测值
        :return:
        '''
        mae = 0
        total = 0

        for j in range(self.data.shape[1]):
            if self.sample_indices[self.user][j] == 0 and self.data[self.user][j] > 0:
                columns = self.train_data[:, j]
                # print('before:', len(columns), end=',')
                columns = columns[columns > 0]
                # print('after:', len(columns))

                if len(columns) > 0:
                    value = np.average(columns)
                else:
                    value = 0

                mae += np.abs(value - self.data[self.user][j])
                total += 1

        return mae / total

    def predict_with_popularity(self):
        '''
        采用求平均值的方式预测值
        :return:
        '''
        mae = 0
        total = 0

        for j in range(self.data.shape[1]):
            if self.sample_indices[self.user][j] == 0 and self.data[self.user][j] > 0:
                columns = self.train_data[:, j]
                # print('before:', len(columns), end=',')
                columns = columns[columns > 0]
                # print('after:', len(columns))

                if len(columns) > 0:

                    value = np.argmax(np.bincount(columns.astype(np.int64)))
                else:
                    value = 0

                mae += np.abs(value - self.data[self.user][j])
                total += 1

        return mae / total

    def predict_with_random(self):
        '''
        采用求平均值的方式预测值
        :return:
        '''
        mae = 0
        total = 0

        for j in range(self.data.shape[1]):
            if self.sample_indices[self.user][j] == 0 and self.data[self.user][j] > 0:
                value = np.random.randint(0, 6)

                mae += np.abs(value - self.data[self.user][j])
                total += 1

        return mae / total

def testSimilarities():
    handler = MovieLensHandler()
    # 装载数据先
    begin = time.time()
    data = handler.readRatings()
    test_data = data[0:3000]

    recommender = EnemyRecommender(test_data, np.random.randint(0, 1000))

    begin = time.time()
    recommender.similarities_test(recommender.train_data[recommender.user])
    print('matrix method cost ', time.time() - begin, 's')

    begin = time.time()
    recommender.similarities_test2(recommender.train_data[recommender.user])
    print('common method cost ', time.time() - begin, 's')

def read_rt_matrix():
    rt_data = []
    with open('../../datasets/ws-dream/rtMatrix.txt', 'r') as f:
        for line in f:
            values = [float(x) for x in line.split()]
            rt_data.append(values)

    return np.array(rt_data)

def test():
    handler = MovieLensHandler()
    #装载数据先
    begin = time.time()
    data = handler.readRatings()
    print(data.shape)
    # data = read_rt_matrix()
    print('data loaded, cost ', time.time() - begin, 's')

    for p in [8, 10]:
        for c in [6, 8, 10]:
            all_maes = []
            all_times = []
            for i in range(10):
                test_data = data
                maes = []
                times = []


                recommender = EnemyRecommender(test_data, pool_size=p, hash_count=c)
                # print('sample train and test data cost ', time.time() - begin, 's')
                #用皮尔逊相似度计算方法计算测试样本中的每个数据与训练集中每个用户的相似度
                # recommender.compute_similarities()

                #计算训练集中所有样本的local sensitive hash值
                begin = time.time()
                recommender.compute_lsh()
                # print('compute lsh cost ', time.time() - begin, "s")

                # print('mae', end=':')
                # begin = time.time()
                # maes.append(recommender.predict_with_pearson())
                # times.append(time.time() - begin)

                begin = time.time()
                maes.append(recommender.predict())
                times.append(time.time() - begin)

                # begin = time.time()
                # maes.append(recommender.predict_with_average())
                # times.append(time.time() - begin)
                #
                # begin = time.time()
                # maes.append(recommender.predict_with_popularity())
                # times.append(time.time() - begin)
                #
                # begin = time.time()
                # maes.append(recommender.predict_with_random())
                # times.append(time.time() - begin)

                all_maes.append(maes)
                all_times.append(times)

            print("===================maes（pool_size=", p, 'hash_count=', c, ')====================')
            for maes in all_maes:
                for m in maes:
                    print(m, end = '\t')
                print('')

            print('=====================times（pool_size=', p, 'hash_count=', c, ')=========================')
            for times in all_times:
                for t in times:
                    print(t, end = '\t')
                print('')
    # for i in range(5):
    #     print(len(recommender.find_enemies_of_enemies(recommender.test_data[i])))

    #计算测试集中用户的敌人的敌人用户，需要先删除相似用户
    # for i in range(20):
    #     enemies = recommender.find_enemies(i)
    #     enemies_enemies = set()
    #     for u in enemies:
    #         u_enemies = recommender.find_enemies(u)
    #         enemies_enemies.union(u_enemies)

# recommender.compute_lsh()
# print('lsh computed!')
# for i in range(100):
#     similar_users = recommender.find_similar_users(data[3000 + i])
#     enemies = recommender.find_enemies(data[3000 + i])
#     # print(len(enemies))
#     enemies_enemies = set()
#     for u in enemies:
#         u_enemies = recommender.find_enemies(recommender.data[u])
#         enemies_enemies = enemies_enemies.union(u_enemies)
#     intersection = set(similar_users) & set(enemies_enemies)
#     print(len(similar_users), len(intersection), len(enemies_enemies))
# begin = time.time()
# similarities = recommender.compute_similarities()
# print('compute similarities cost ', time.time() - begin, "s")
#
# recommender.compute_lsh()
# for i in range(20):
#     print(len(recommender.find_similar_users(i)), end=',')
#     print(len(recommender.find_similar_users(i, filter=True)))

test()