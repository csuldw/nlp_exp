# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 23:57:22 2018

@author: liudiwei
"""
import jieba

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
            



import json

#获取数据集
def generate_corpus(input_file):
    with open(input_file , "r", encoding='UTF-8') as f: 
        train_data = json.load(f)
    train_sentences = train_data["sentences"]
    corpus_english = set() #0
    corpus_spanish = set() #1
    corpus_chinese = set() #2
    for sentence in train_sentences:
        text = sentence["text"].replace("\n", "")
        language_label = sentence["language_label"]
        if 0 == language_label:
            seg = text.lower().split(" ")
            corpus_english.union(seg)
        elif 1 == language_label:
            corpus_spanish.union(jieba.lcut(text))
        elif 2 == language_label:
            corpus_chinese.union(jieba.lcut(text))
    return corpus_english, corpus_spanish, corpus_chinese

