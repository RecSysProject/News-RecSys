#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import _pickle as cPickle
from datetime import datetime
from gensim.models import Word2Vec
warnings.filterwarnings('ignore')

class deepWalk(object):
    def __init__(self):
        self.root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

        self.newsSet = {}            #新闻集
        self.userSet = {}            #用户集
    
        self.dataDf=pd.read_csv(self.root+'dataset_CF.csv')
        self.df = self.dataDf[self.dataDf['label'].isin([1])].astype(str)
        
        # construct an undirected graph
        self.path_length = 10
        self.G = nx.from_pandas_edgelist(self.df, "user_id", "new_id", edge_attr=True, create_using=nx.Graph())
        self.random_walks = []
        print("输出全部节点的数量：{}".format(len(self.G)))
        print("输出全部边的数量：{}".format(self.G.number_of_edges()))


    'function to generate random walk sequences of nodes'
    def get_randomwalk(self, node, path_length):   # 将节点和被遍历的路径的长度作为输入
        
        random_walk = [node]
        '它将从指定的输入节点以随机的方式穿过连接节点。最后,返回遍历节点的顺序'
        for i in range(path_length-1):
            temp = list(self.G.neighbors(node))
            temp = list(set(temp) - set(random_walk))
            if len(temp) == 0:
                break
            
            random_node = random.choice(temp)
            random_walk.append(random_node)
            node = random_node
        
        return random_walk
    
    def prepared(self):
        
        for new in self.df.new_id.unique():
            self.newsSet[new] = 1

        for user in self.df.user_id.unique():
            self.userSet[user] = 1

        #捕获数据集中所有节点的随机游走序列
        # 从图获取所有节点的列表
        all_nodes = list(self.G.nodes())

        for n in tqdm(all_nodes):
            for i in range(5):
                self.random_walks.append(self.get_randomwalk(n, self.path_length))

        # count of sequences 序列个数
        print("输出全部随机序列的数量：{}".format(len(self.random_walks)))

    # Word2Vec skip-gram
    def word2vec_embedding(self):
        # train word2vec model 使用随机游走训练skip-gram模型
        model = Word2Vec(size = 128, window = 4, sg = 1, hs = 0,
                         negative = 10, # for negative sampling
                         alpha=0.03, min_alpha=0.0007,
                         seed = 14)

        model.build_vocab(self.random_walks, progress_per=2)

        model.train(self.random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)

        print(model) #图中的每个节点都由固定长度（128）的向量表示

        # news embedding矩阵
        for news, num in self.newsSet.items():
            self.newsSet[news] = model[news]

        cPickle.dump(self.newsSet, open(self.root + 'multi_news_graph.pkl', 'wb'))

        # user embedding矩阵
        for user, num in self.userSet.items():
            self.userSet[user] = model[user]

        cPickle.dump(self.userSet, open(self.root + 'multi_user_graph.pkl', 'wb'))

        # binary_user_feature
        user_feature = []
        for user in self.dataDf['user_id']:
            user_feature.append(self.userSet[str(user)])

        cPickle.dump(user_feature, open(self.root + 'binary_user_graph.pkl', 'wb'))

        # binary_news_feature
        news_feature = []
        for news in self.dataDf['new_id']:
            news_feature.append(self.newsSet[str(news)])

        cPickle.dump(news_feature, open(self.root + 'binary_news_graph.pkl', 'wb'))

if __name__ == '__main__':
    start = datetime.now()
    test = deepWalk()
    test.prepared()
    test.word2vec_embedding()
    print("This took ", datetime.now() - start)



