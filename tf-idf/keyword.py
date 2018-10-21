# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:35:24 2018

@author: liudiwei
"""

import codecs
from gensim.models import Word2Vec
import jieba.posseg as psg
import collections
import math


#判断给定字符（串）是否为汉字，返回True或False
def isChinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

"""
利用jieba切词（只保留中文词语），传入一段中文文本，结果返回该文本的切词结果（列表形式）
"""
def cutWords(eachText):
    #stopList = []
    #for stopWord in codecs.open('F:/getKeyWords/stopwords.txt', 'r', 'utf-8'):
    #    stopList.append(stopWord.strip())
    words = psg.cut(eachText)
    wordsList = []
    for oneWord in words:
        flag = True
        for i in range(len(oneWord.word)):
            if not isChinese(oneWord.word[i]):
                flag = False
                break
        if flag and len(oneWord.word) > 1:
            wordsList.append(oneWord.word)
    return wordsList


"""
计算Inverse Doucument Frequency  
入参：word(单词)、docList(文档列表)
返回:idf
"""
def idf(word, docList):
    docNumWithTerm = sum(1 for doc in docList if word in doc)
    docNum = len(docList)  #语料库的文档总数
    return math.log2(float(docNum) / float(docNumWithTerm + 1))

"""
传入文档路径（每行为一篇文档），计算每篇文本中每个词的tfidf值
输出形式如下：
[[文本编号1，文本1的词语1，该词tfidf值],[文本编号2，文本2的词语1，该词tfidf值]……]
"""
def getTFIDF(filePath):
    docList = codecs.open(filePath, 'r', 'utf-8').readlines()
    
    docIndex = 0    #文档索引
    tfidf = []      #tfidf列表    
    for eachLine1 in codecs.open(filePath, 'r', 'utf-8'):
        docIndex += 1
        lineList = cutWords(eachLine1)
        #计算一个词的tf值
        wordsNum = len(lineList)    #当前文本的总词数
        tempDict0 = collections.Counter(lineList)
        wordList = list(tempDict0.keys())
        for i in range(len(wordList)):
            tfidfItem = []
            word = wordList[i]
            wordTF = float(tempDict0[word]) / float(wordsNum)
            
            #计算 Inverse Doucument Frequency
            wordIDF = idf(word, docList)
            
            tfidfItem.append(docIndex)
            tfidfItem.append(word)
            tfidfItem.append(wordTF * wordIDF)
            tfidf.append(tfidfItem)
    return tfidf

"""
获取文章的关键字
根据文章tfidf矩阵获取文章的关键字
输入参数：
    tfidfList: [[文本编号1，文本1的词语1，该词tfidf值],[文本编号2，文本2的词语1，该词tfidf值]……]
    docIds: 文档ID
    k: 关键字的数量
输出格式：
    第一列是文档ID，其他列是keyword
"""
def getKeywords(tfidfList, docIds=[], k=10):
    doc2Keyword = []
    for docId in docIds:
        keywordItem = []
        #获取第docId文档的tfidf矩阵
        tfidfSub = [ item for item in tfidfList if item[0] == docId]
        #根据tfidf值进行逆序排序
        sortedTFIDF = sorted(tfidfSub, key = lambda x:x[2], reverse=True)
        #获取tfidf前k个keyword
        topKWord = [sortedTFIDF[i][1] for i in range(len(sortedTFIDF)) if i < k]
        
        keywordItem.append(docId)
        keywordItem.append("|".join(topKWord))
        doc2Keyword.append(keywordItem)   
    return doc2Keyword

if __name__ == "__main__":
   tfidf = getTFIDF("document.data")
   docIds = list(set([elem[0] for elem in tfidf ]))
   doc2keyword = getKeywords(tfidf, docIds, 100)