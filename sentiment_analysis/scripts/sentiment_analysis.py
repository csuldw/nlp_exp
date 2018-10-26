# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:34:06 2018

@author: liudiwei
"""

import jieba
from nltk.collocations import  BigramCollocationFinder
from nltk.metrics import  BigramAssocMeasures
import json
import jieba.analyse
from random import shuffle
from nltk.probability import  FreqDist,ConditionalFreqDist
from nltk.classify.scikitlearn import  SklearnClassifier
from sklearn.svm import SVC, LinearSVC,  NuSVC
from sklearn.naive_bayes import  MultinomialNB, BernoulliNB
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import  accuracy_score
from langdetect import detect

class SentimentAnalysis():
    
    def __init__(self, stopwords=[]):
        self.stopwords = stopwords
        self.classifier = None

    #数据清洗
    def _filter(self, text):    
        words = text.split(" ")
        words_filter= []
        for word in words:
            #过滤掉带有@符号的昵称 AND 去停留词
            if "@" in word or word in self.stopwords:
                continue
            words_filter.append(word)
            
        return " ".join(words_filter)
    
    
    #分词获取tag
    def _get_tags(self, text):
        #tags = jieba.analyse.extract_tags(text, topK=100, withWeight=True, allowPOS=('n', 'nv')) 
        tags = jieba.analyse.extract_tags(text, topK=100, withWeight=True) 
        return dict((k, True) for k,v in tags)
    
    #获取数据集
    def _build_train_data(self, input_file):
        with open(input_file , "r", encoding='UTF-8') as f: 
            train_data = json.load(f)
        train_sentences = train_data["sentences"]
        anger_data = []
        fear_data = []
        joy_data = []
        sadness_data = []
        for sentence in train_sentences:
            text = sentence["text"]
            emotion_label = sentence["emotion_label"]
            #language_label = sentence["language_label"]
            tags = self._get_tags(self._filter(text))
            word_list = []
            if 0 == emotion_label:
                word_list.append(tags)
                word_list.append(emotion_label)
                anger_data.append(word_list)
            elif 1 == emotion_label:
                word_list.append(tags)
                word_list.append(emotion_label)
                fear_data.append(word_list)
            elif 2 == emotion_label:
                word_list.append(tags)
                word_list.append(emotion_label)
                joy_data.append(word_list)
            elif 3 == emotion_label:
                word_list.append(tags)
                word_list.append(emotion_label)
                sadness_data.append(word_list)
        return anger_data, fear_data, joy_data, sadness_data
    
    #训练数据
    def train(self, train, classifier):
        classifier = SklearnClassifier(classifier) 
        classifier.train(train) #训练分类器
        self.classifier = classifier
        return self
    
    #预测
    def predict_many(self, test_dataset):
        predicted_label = self.classifier.classify_many(test_dataset) #给出预测的标签
        return predicted_label
    
    #计算精度[]
    def acc_score(self, predicted, labels):
        n = 0
        s = len(predicted)
        for i in range(0,s):
            if predicted[i] == labels[i]:
                n = n+1
        return n/s #分类器准确度 
    
    #建立测试集
    def _build_test_data(self, test_file):
        with open(input_file , "r", encoding='UTF-8') as f: 
                row_data = json.load(f)
        #info = row_data["info"]
        sentences = row_data["sentences"]
        
        sentences_lang = []
        test_sentences = []
        for sentence in sentences:
            text = sentence["text"]
            tags = self._get_tags(self._filter(text))
            lang_type = detect(text)
            #使用langdetect进行语言检测
            if lang_type == 'zh-cn':
                sentence["language_label"] = 2
            elif lang_type == 'en':
                sentence["language_label"] = 0
            elif lang_type == '':
                sentence["language_label"] = 1
            else:
                print("language detect error: ", text)
            test_sentences.append(tags)
            sentences_lang.append(sentence)
            
        row_data["sentences"] = sentences_lang 
        return row_data, test_sentences
    
    #将数据模拟成训练集的格式，便于提交代码
    def simulation_data(self, row_data, predict_label):
        sentences = row_data["sentences"]
        for i in range(len(sentences)):
            sentences[i]["emotion_label"] = predict_label
        row_data["sentences"] = sentences
        return row_data
            
#3.训练模型
if __name__ == "__main__":
    stopwords = [line.strip() for line in open("stopwords.txt", 'r', encoding='UTF-8')]
   
    clf = SentimentAnalysis(stopwords=stopwords)
    input_file = "../data/train.json"
    anger_data, fear_data, joy_data, sadness_data = clf._build_train_data(input_file)
    
    shuffle(anger_data) 
    shuffle(fear_data) #把文本的排列随机化  
    train =  anger_data + joy_data      #训练集
    test = joy_data + sadness_data      #验证集    
    data,tag = zip(*test)
    
    #MultinomialNB() LogisticRegression  LinearSVC NuSVC 
    clf = clf.train(train, BernoulliNB())
    predicted_label = clf.predict_many(data)
    print("预测类别为:", predicted_label)
    print("真是类别为", tag)
    score =  clf.acc_score(predicted_label, tag)
    
    print('accuracy is %f'  %score)
    
    row_data, test_sentences = clf._build_test_data("../data/test.json")
    test_sentences = tuple(test_sentences)
    
    test_predict = clf.predict_many(test_sentences)
    
    predict_result = clf.simulation_data(row_data, test_predict)