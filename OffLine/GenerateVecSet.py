#!/usr/bin/env python

import sys
import jieba
import numpy as np
import pandas as pd
import _pickle as cPickle
from datetime import datetime
from operator import itemgetter
from gensim.models import word2vec

class GenerateVecSet(object):
    def __init__(self):
        #行为序列 新闻集合 positive
        self.newsSet = {}            #所有新闻样本
        self.newsSet_count = 0       #新闻计数
        
        self.news_popular = {}       #item ：流行度
        self.UserSequence = {}       #用户行为序列
        self.UserSequence_len = 0    #用户行为总条数
        self.userTimeContext = {}    #用户行为序列矩阵
        
        self.posSet = {}    #正样本
        self.dataSet = {}   #总体样本
        self.negSet = {}    #负样本
        
        self.root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'
        
        self.word_model = word2vec.Word2Vec.load('/Users/tung/Python/PersonalProject/NewsRecommend/word-embedding/sougouCS_wordVec')                #加载已训练好的word2vec模型
        self.vocab = self.word_model.wv.vocab.keys()    # 先拿到全部的vocabulary

    def split_dataset(self):
        #导入行为数据,生成正样本
        f1=open(self.root+'data/train_date_set1.txt')

        label = 1
        posSet_len = 0
        dataSet_len = 0

        "遍历所有浏览记录，保存所有用户浏览过的新闻id到字典，返回{用户id：news_id1, news_id2…}"

        for line in f1.readlines():
            user = line.strip().split('\t')[0]
            news = line.strip().split('\t')[1]
            time = line.strip().split('\t')[2]
            title = line.strip().split('\t')[3]
            
            self.UserSequence.setdefault(user, {})
            self.UserSequence[user][news] = time
            
            # 统计新闻个数
            if news not in self.newsSet:
                self.newsSet[news] = title
                self.newsSet_count += 1
            
            self.UserSequence_len += 1
            
            self.posSet.setdefault(user, {})
            self.dataSet.setdefault(user, {})
            
            self.posSet[user][news] = label
            self.dataSet[user][news] = label
            posSet_len += 1
            dataSet_len += 1

        print ('UserSequence %s rowCount' % self.UserSequence_len, file=sys.stderr)
        print('Total news number = %d' % self.newsSet_count, file=sys.stderr)
        # 新闻title持久化
        cPickle.dump(self.newsSet, open(self.root+'newsSet_title.pkl', 'wb'))

        #negative
        #保证正负样本均衡1:1
        #取用户没有阅读过的，热度比较高的news为负样本

        #统计热度
        news_count = 0
        for user, news in self.posSet.items():
            for new in news:
                # 统计 item 流行度
                if new not in self.news_popular:
                    self.news_popular[new] = 0
                self.news_popular[new] += 1

        print('count news number and popularity succ', file=sys.stderr)

        # save the total number of news
        news_count = len(self.news_popular)
        print('total news number = %d' % news_count, file=sys.stderr)

        #打负标签
        label = 0
        negSet_len = 0

        for user, news in self.posSet.items():
            watched_news = self.posSet[user]
            postive_len = len(news)
            for news, popular in sorted(self.news_popular.items(), key=itemgetter(1), reverse=True)[0:2*postive_len]:
                if news in watched_news:  #去除用户已看过的
                    continue
                if postive_len > 0 :
                    self.negSet.setdefault(user, {})
                    self.negSet[user][news] = label
                    self.dataSet[user][news] = label
                    
                    postive_len -= 1
                    negSet_len += 1
                    dataSet_len += 1

        # dataSet_len = posSet_len+negSet_len
        print ('posSet_len %s generate' % posSet_len, file=sys.stderr)
        print ('negSet_len %s generate' % negSet_len, file=sys.stderr)
        print ('dataSet_len %s generate' % dataSet_len, file=sys.stderr)

        #创建正负样本文件
        f=open(self.root+'dataset_Vec.csv','w')
        ocolnames = ['user_id', 'new_id', 'label']
        f.write( ','.join(ocolnames) + '\n' )

        for user, news in self.dataSet.items():
            for new, label in news.items():
                f.write( str(user) +','+ str(new) + ',' + str(label) + '\n')
        f.close()

    #news embedding
    #分词
    def get_stop_words(self):
        path = "/Users/tung/Python/PersonalProject/NewsRecommend/word-embedding/chineseStopWords.txt"
        file = open(path, 'rb').read().decode('utf-8').split('\r\n')
        return set(file)

    def rm_stop_words(self, word_list):
        word_list = list(word_list)
        stop_words = self.get_stop_words()
        # 这个很重要，注意每次pop之后总长度是变化的
        for i in range(word_list.__len__())[::-1]:
            # 去停用词
            if word_list[i] in stop_words:
                word_list.pop(i)
            #  去数字
            elif word_list[i].isdigit():
                word_list.pop(i)
        return word_list

    def cut_candidate(self, candidate):
        out = []
        for i in candidate:
            temp = jieba.cut(str(i), cut_all=False)
            # 去停用词
            temp = self.rm_stop_words(temp)
            out.append( ' '.join(temp))
        return out

    #向量化
    # 得到任意text的vector
    def get_vector(self, word_list):
        
        #所有分词的vector取均值
        #建立一个全是0的array
        res =np.zeros([128])
        count = 0
        for word in word_list:
            if word in self.vocab:
                res += self.word_model[word]
                count += 1
        return res/count if count >0 else res

    #news embedding矩阵
    def news_embedding(self):
        
        for news, title in self.newsSet.items():
            cut = self.cut_candidate(title)
            vector = self.get_vector(cut)
            self.newsSet[news] = vector       #字典中news的title更新为vector

        print('news matrix length', len(self.newsSet), file=sys.stderr)

        #持久化multi_newsSet2Vec
        #新闻对应的embedding矩阵
        newsSet2Vec = {}
        for news, vector in self.newsSet.items():
            if news not in newsSet2Vec:
                newsSet2Vec[news] = vector

        cPickle.dump(newsSet2Vec, open(self.root+'multi_news_word.pkl', 'wb'))

    #行为序列embedding
    def user_embedding(self):
        
        Context_len = 3
        #取前Context_len条embedding
        for user, news in self.UserSequence.items():
            temp = []
            Context_len = 3      # 取的用户序列长度
            self.userTimeContext.setdefault(user)
            
            for new in news:
                Context_len -= 1
                if Context_len < 0:
                    continue
                temp += list(self.newsSet[new])   #序列中news的vector相加
            
            while len(temp) < 128*3:   # sen没那么长,直接贴上全是0的vec
                temp += [0] * 128
            
            self.userTimeContext[user] = np.array(temp)

        print('userTimeContext length', len(self.userTimeContext), file=sys.stderr)

        #持久化multi_user_profile
        #用户对应的行为序列embedding矩阵

        user_profile = {}
        for user, vector in self.userTimeContext.items():
            if user not in user_profile:
                user_profile[user] = vector

        cPickle.dump(user_profile, open(self.root + 'multi_user_word.pkl', 'wb'))
    
    def Generate_Set(self):
        #持久化multi_user_label
        #多分类 一个用户对应一个label
        user_lable = np.zeros((len(self.UserSequence),self.newsSet_count))# 共user_id行 每行newsSet_count列 初值为0
        
        row = 0
        for user, news in self.UserSequence.items():
            col = 0
            temp = [0]*self.newsSet_count
            
            for new in self.newsSet:
                if new in news:
                    temp[col] = 1
                else:
                    temp[col] = 0
                col += 1
    
        user_lable[row] = temp
        row += 1
        
        print('user_lable shape', user_lable.shape, file=sys.stderr)
        cPickle.dump(user_lable, open(self.root + 'multi_user_lable.pkl', 'wb'))
        
        #持久化binary_news_feature
        news_feature = []
        for user, news in self.dataSet.items():
            for new, label in news.items():
                news_feature.append(self.newsSet[new])

        cPickle.dump(news_feature, open(self.root + 'binary_news_word.pkl', 'wb'))   #新闻embedding字典

        #持久化binary_user_feature
        user_feature = []
        for user, news in self.dataSet.items():
            for new, label in news.items():
                user_feature.append(self.userTimeContext[user])

        cPickle.dump(user_feature, open(self.root + 'binary_user_word.pkl', 'wb'))   #对应的embedding矩阵

        #持久化binary_lable
        df=pd.read_csv(self.root+'dataset_Vec.csv')
        cPickle.dump(df.label, open(self.root + 'binary_label.pkl', 'wb'))   #对应的embedding矩阵

if __name__ == '__main__':
    start = datetime.now()
    test = GenerateVecSet()
    test.split_dataset()
    test.news_embedding()
    test.user_embedding()
    test.Generate_Set()
    print("This took ", datetime.now() - start)
