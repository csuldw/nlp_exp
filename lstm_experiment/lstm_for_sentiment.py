# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:55:51 2018

@author: liudiwei
"""

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


#加载原始数据集
def build_corpus():
    print("build corpus.")
    pos_data = pd.read_excel('./data/pos.xls', header=None, index=None) #读取训练语料完毕
    neg_data = pd.read_excel('./data/neg.xls', header=None, index=None)

    #给训练语料贴上标签
    pos_data['label']=1
    neg_data['label']=0 
    raw_corpus = pd.concat([pos_data,neg_data], ignore_index=True) #合并语料
    
    #计算语料数目
    pos_len=len(pos_data) 
    neg_len=len(neg_data)
    print("pos size: ", pos_len, ", neg size: ", neg_len)
    return raw_corpus, pos_len, neg_len
 

def build_dataset(raw_corpus, pos_len, neg_len, maxlen):
    #分词
    print("start to cut words")
    cut_words = lambda x: list(jieba.cut(x)) #定义分词函数
    raw_corpus['words'] = raw_corpus[0].apply(cut_words)
    d2v_train = raw_corpus['words']
    
    #combine all word to list
    print("combine all word to list")
    word_list = [] 
    for i in d2v_train:
      word_list.extend(i)
    
    #word value count
    print("word value count")
    word_count = pd.DataFrame(pd.Series(word_list).value_counts()) 
    print("free memory")
    del word_list,d2v_train
    
    # add index to word_count
    print("add index to word_count")
    word_count['id']=list(range(1, len(word_count) + 1))
    
    #将word的id对应到raw_corpus
    print("get id sentence")
    get_sent = lambda x: list(word_count['id'][x])
    raw_corpus['sentence'] = raw_corpus['words'].apply(get_sent) #速度太慢
    
    raw_corpus['sentence'] = list(sequence.pad_sequences(raw_corpus['sentence'], maxlen=maxlen))
    
    #train dataset
    print("get train dataset and test dataset.")
    train_x = np.array(list(raw_corpus['sentence']))[::2] 
    train_y = np.array(list(raw_corpus['label']))[::2]
    
    #test dataset
    test_x = np.array(list(raw_corpus['sentence']))[1::2] #测试集
    test_y = np.array(list(raw_corpus['label']))[1::2]
    
    ##全集
    x_all = np.array(list(raw_corpus['sentence'])) 
    y_all = np.array(list(raw_corpus['label']))
    
    return train_x, train_y, test_x, test_y, x_all, y_all, word_count

def train_lstm(train_x, train_y, word_count, batch_size, epochs):
    print('Start to build model...')
    model = Sequential()
    model.add(Embedding(len(word_count)+1, 256))
    model.add(LSTM(256))        # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=128, output_dim=1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    #start to fit train
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs) 
    return model

    
if __name__ == "__main__":
    raw_corpus, pos_len, neg_len = build_corpus()  
    maxlen = 150 #序列的最大长度
    batch_size = 32
    epochs = 5
    
    train_x, train_y, test_x, test_y, x_all, y_all, word_count = build_dataset(raw_corpus, pos_len, neg_len, 150)
    model = train_lstm(train_x, 
                       train_y, 
                       word_count,
                       batch_size,
                       epochs)
    classes = model.predict_classes(test_x)
    loss = model.evaluate(test_x,test_y) #模型测试
    print('Test loss:', loss)
    
    num = 0
    for i in range(len(classes)):
        if classes[i] == test_y[i]:
            num += 1
    print('Test Acc:', float(num/len(classes)))