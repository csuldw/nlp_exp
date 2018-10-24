# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:57:01 2018

@author: liudiwei
"""

import pandas as pd
import math
import operator


#相似度计算

#user与item的评分关系
"""
输入：DataFrame
"""
def user_item_relation(train_data):
    user_items = dict()
    for index, row in train_data.iterrows():
        if row[0] not in user_items.keys():
            user_items[row[0]] = []
        user_items[row[0]].append((row[1], row[2]))
    return user_items

#item与user的倒排评分关系
def item_users_relation(train_data):
    item_users = dict()
    for index, row in train_data.iterrows():
        if row[1] not in item_users.keys():
            item_users[row[1]] = []
        item_users[row[1]].append((row[0], row[2]))
    return item_users
    
"""user与user的相似度计算
输入：输入为item与user的对应关系 {item_id: [(user_id, score1), (user_id2, score2)]...}
输出:用户相似矩阵 {user_id: {user_id: score}, user_id: {user_id: score}, ...}
"""
def user_similarity(item_users):
    C = dict()
    N = dict()
    
    # calculate co-related items between users
    for i, user_score in item_users.items():
        for u, score in user_score:
            if u not in N.keys():
                N[u] = 0
            N[u] += 1
            for v, score in user_score:
                if u == v:
                    continue
                if u not in C.keys():
                    C[u] = dict()
                if v not in C[u].keys():
                    C[u][v] = 0
                C[u][v] = 1
                
    # calculate user similarity matrix
    similarity_matrix = dict()
    for u, related_user in C.items():
        for v, cuv in related_user.items():
            if u not in similarity_matrix.keys():
                similarity_matrix[u] = dict()
            if v not in similarity_matrix[u].keys():
                similarity_matrix[u][v] = 0
            similarity_matrix[u][v] = cuv/math.sqrt(N[u] * N[v])
    return similarity_matrix


def recommand(user, train_data, similarity_matrix, k):
    rank = dict()
    #获取user与item的关系
    user_items = user_item_relation(train_data)
    
    #获取相似程度最大的k个用户
    related_users = sorted(similarity_matrix[user].items(), \
                           key=operator.itemgetter(1), reverse=True)[0:k]
    
    #遍历相似的用户
    for (v, wscore) in related_users:
        #遍历各个用户的item
        for (i, viscore) in user_items[v]:
            if i in [x[0] for x in user_items[user]]:
                continue
            rank[i] = wscore * viscore
    
    #对结果进行排序
    rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
    return rank

"""train.csv
input data
user_id, item_id, score
"""
if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    item_users = item_users_relation(train_data)
    user_similarity_matrix = user_similarity(item_users)
    rank = recommand("u1004", train_data, user_similarity_matrix, 3)