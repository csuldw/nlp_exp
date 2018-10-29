# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 23:57:22 2018

@author: liudiwei
"""

#将数据压缩
def scaling_data(pos_data, neg_data):
    pos_len = len(pos_data)
    neg_len = len(neg_data)
    if pos_len > neg_len:
        pos_new = pos_data[:neg_len]
        pos_other = pos_data[neg_len:]
        for i in range(len(pos_other)):
            pos_new[i%neg_len][0] += pos_other[i][0]
        pos_data = pos_new
    if neg_len > pos_len:
        neg_new = pos_data[:neg_len]
        neg_other = pos_data[neg_len:]
        for i in range(len(pos_other)):
            neg_new[i%neg_len][0] += neg_other[i][0]    
        neg_data = neg_new
    return pos_data, neg_data
            
    