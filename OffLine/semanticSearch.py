import sys
import faiss
import jieba
import logging
import numpy as np
import _pickle as cPickle
from datetime import datetime
from jieba import analyse
from gensim.models import word2vec

jieba.setLogLevel(logging.INFO)

class search(object):
    def __init__(self):
        
        self.root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'
        self.newsSet = cPickle.load( open(self.root + 'newsSet_title.pkl','rb') )               # 新闻title字典
        
        '加载word2vec模型'
        self.word_model = word2vec.Word2Vec.load('/Users/tung/Python/PersonalProject/NewsRecommend/word-embedding/sougouCS_wordVec')
        self.vocab = self.word_model.wv.vocab.keys()
        
        self.word_news_feature = cPickle.load( open(self.root + 'multi_news_word.pkl','rb') )   # 新闻词嵌入
        self.word_news_feature_id = list(self.word_news_feature.keys())
        self.word_news_feature_vec = np.array(list(self.word_news_feature.values())).astype('float32')
    
    # 得到任意text的vector
    def get_vector(self, word_list):
        # 建立一个全是0的array
        res =np.zeros([128])
        count = 0
        for word in word_list:
            if word in self.vocab:
                res += self.word_model[word]
                count += 1
        return res/count if count >0 else res

    def FlatL2(self, query):
        # Query扩展
        tags = analyse.extract_tags(query,5)
        
        queryVec = self.get_vector(tags)
        queryVec = queryVec.reshape(1,128).astype('float32')
        
        #相似度检索
        dim = 128         # 向量维度
        k = 10            # 召回向量个数
        index = faiss.IndexFlatL2(dim)         # L2距离，欧式距离（越小越好）
        index.add(self.word_news_feature_vec)  # 添加训练时的样本
        D, I = index.search(queryVec, k)       # 寻找相似向量， I表示相似向量ID矩阵， D表示距离矩阵

        res = []
        for i in I[0]:
            news_id = self.word_news_feature_id[i]
            res.append((news_id, self.newsSet[news_id]))

        return res
    
    def IVFFlat(self, query):
        # Query扩展
        tags = analyse.extract_tags(query,5)
        
        queryVec = self.get_vector(tags)
        queryVec = queryVec.reshape(1,128).astype('float32')
        
        #相似度检索
        dim = 128         # 向量维度
        k = 10            # 定义召回向量个数
        nlist = 100       #聚类中心的个数
        quantizer = faiss.IndexFlatL2(dim)         # 定义量化器
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.nprobe = 10                          #查找聚类中心的个数，默认为1个，若nprobe=nlist则等同于精确查找
        assert not index.is_trained
        index.train(self.word_news_feature_vec)    #需要训练
        assert index.is_trained
        index.add(self.word_news_feature_vec)      #添加训练时的样本
        D, I = index.search(queryVec, k)           #寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵

        res = []
        for i in I[0]:
            news_id = self.word_news_feature_id[i]
            res.append((news_id, self.newsSet[news_id]))
        
        return res
    
    def factory(self, query):
        # Query扩展
        tags = analyse.extract_tags(query,5)
        
        queryVec = self.get_vector(tags)
        queryVec = queryVec.reshape(1,128).astype('float32')
        
        #相似度检索
        dim = 128         # 向量维度
        k = 10            # 召回向量个数
        index = faiss.index_factory(dim, "PCAR32,IVF100,SQ8") #PCA降到32位；搜索空间100；SQ8,scalar标量化，每个向量编码为8bit(1字节)
        assert not index.is_trained
        index.train(self.word_news_feature_vec)    #需要训练
        assert index.is_trained
        index.add(self.word_news_feature_vec)      #添加训练时的样本
        D, I = index.search(queryVec, k)           #寻找相似向量， I表示相似用户ID矩阵， D表示距离矩阵
        
        res = []
        for i in I[0]:
            news_id = self.word_news_feature_id[i]
            res.append((news_id, self.newsSet[news_id]))
        
        return res

if __name__ == '__main__':
    start = datetime.now()
#    query = '经济'
    query = sys.argv[1]
    
    test = search()
    result = test.FlatL2(query)

    print('input query:%s' % query)
    for news_id, title in result:
        print('id:%s, 标题:%s' % (news_id, title))
    print("This took ", datetime.now() - start)


