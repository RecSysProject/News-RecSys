#!/usr/bin/env python

import re
import os
import jieba
import gensim
import logging
import numpy as np
from datetime import datetime
from gensim.models import word2vec

class wordEmbedding(object):
    def __init__(self):
        self.root = '/Users/tung/Python/PersonalProject/NewsRecommend/word-embedding/'
        self.train_file_name = self.root + "sougouCS_corpus.txt"
        self.save_model_name = self.root + 'sougouCS_wordVec'
    
    '去停用词'
    def get_stop_words(self):
        path = self.root + "chineseStopWords.txt"
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

    '去低频词'
    def rm_word_freq_so_little(self, dictionary, freq_thred):
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < freq_thred ]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()

    '生成原始语料文件夹下文件列表'
    def listdir(self, path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            else:
                list_name.append(file_path)
    '生成语料'
    def prepareCorpus(self):
        list_name = []
        self.listdir(self.root+'SogouCS.reduced/', list_name)

        '''字符数小于这个数目的content将不被保存'''
        threh = 30
        dictionary = []
        for path in list_name:
            print(path)
            
            file = open(path, 'rb').read().decode('utf-8', 'ignore')
            
            '''正则匹配出contenttitle与content'''
            patternCtt = re.compile(r'<content>(.*?)</content>', re.S)
            contents = patternCtt.findall(file)
            
            patternCtte = re.compile(r'<contenttitle>(.*?)</contenttitle>', re.S)
            contenttitle = patternCtte.findall(file)
            
            '''contenttitle与content合并'''
            for i in range(contents.__len__()):
                contents[i] = contenttitle[i] +'\n' + contents[i]

            '''把所有内容小于30字符的文本全部过滤掉'''
            for i in range(contents.__len__())[::-1]:
                if len(contents[i]) < threh:
                    contents.pop(i)

            for text in contents:
                content = text
                word_list = list(jieba.cut(content, cut_all=False))  # 分词
                word_list = self.rm_stop_words(word_list)            # 去停用词
                
                dictionary.append(' '.join(word_list))

        for i in range(dictionary.__len__()):
            file = self.root + 'sougouCS_corpus.txt'
            f = open(file, 'a+', encoding='utf-8')
            f.write(''.join(dictionary[i]) +'\n')        #加\n换行显示

#        print('语料总新闻条数：', np.shape(dictionary))

    '训练生成词向量'
    def modeling(self):
        if not os.path.exists(self.save_model_name):     # 判断文件是否存在
            
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            sentences = word2vec.Text8Corpus(self.train_file_name, max_sentence_length = 1000)  # 加载语料
            model = gensim.models.Word2Vec(sentences, size=128, window=4, sg = 1, hs = 0,
                                           negative = 10, alpha=0.03, min_alpha=0.0007, seed = 14, min_count=20)
            # 训练skip-gram模型,window=4
            model.save(self.save_model_file)
            model.wv.save_word2vec_format(self.save_model_name+".bin", binary=True) # 二进制类型保存以便重用
        else:
            print('此训练模型已经存在，不用再次训练')

        '加载模型'
        wordVec = word2vec.Word2Vec.load(self.save_model_name)
        # 查看构成的词汇表
        print('词汇表长度', len(wordVec.wv.vocab.keys()))

if __name__ == '__main__':
    start = datetime.now()
    test = wordEmbedding()
    test.prepareCorpus()
    test.modeling()
    print("This took ", datetime.now() - start)

