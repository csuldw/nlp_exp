# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 23:19:26 2018

@author: liudiwei
"""

import codecs
from gensim.models import LdaModel
from gensim import models, corpora

#
def load_data(filePath):
    documents = codecs.open(filePath, 'r', 'utf-8').readlines()
    return documents

def get_texts(documents):
    texts = [doc.lower().split() for doc in documents]
    return texts

"""
输入texts是分词之后的文本内容
"""
def get_corpus_and_word_dict(texts):
    word_dict = corpora.Dictionary(texts)    #自建词典
    
    #通过dict将用字符串表示的文档转换为用id表示的文档向量
    corpus = [word_dict.doc2bow(text) for text in texts]
    return corpus, word_dict 

if __name__ == "__main__":
    documents = load_data("document_en.data")
    
    texts = get_texts(documents)
    #print(texts)
    
    corpus, word_dict = get_corpus_and_word_dict(texts)
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    print(corpus)
    print(corpus_tfidf)
    lda = LdaModel(corpus=corpus_tfidf, id2word=word_dict, num_topics=10)
    doc_topic = [a for a in lda[corpus]]
    
    topics_r = lda.print_topics(num_topics = 10, num_words = 10)
    
    fw = codecs.open('topics_result.txt','w')
    for v in topics_r:
        fw.write(str(v)+'\n')
    fw.close()