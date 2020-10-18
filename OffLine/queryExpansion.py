import re
import os
import sys
import jieba
import gensim
import logging
from datetime import datetime
from gensim.models import word2vec
jieba.setLogLevel(logging.INFO)

class queryExpan(object):
    def __init__(self):
        self.root = '/Users/tung/Python/PersonalProject/RentPlatform/'
        '训练word2vec'
        self.Raw_corpora = self.root + "lianjia/"
        self.train_file = self.root + "rent_corpus.txt"
        self.save_model_name = self.root + "rent_wordVec"
    
        '加载word2vec模型'
        self.word_model = word2vec.Word2Vec.load(self.save_model_name)
        self.vocab = self.word_model.wv.vocab.keys()

    # 预处理
    ## 去停用词
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

    ## 去低频词
    def rm_word_freq_so_little(self, dictionary, freq_thred):
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < freq_thred ]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()

    ## 分词
    '''生成原始语料文件夹下文件列表'''
    def listdir(self, path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.listdir(file_path, list_name)
            else:
                list_name.append(file_path)
        return list_name

    def generate_corpors(self):
        list_name = []
        list_name = self.listdir(self.Raw_corpora, list_name)
        '''字符数小于这个数目的content将不被保存'''
        threh = 2
        
        dictionary = []
        for path in list_name:
            print(path)
            
            file = open(path, 'rb').read().decode('utf-8', 'ignore')
            content = file.split("\r\n")   #'\r'是回车，'\n'是换行，前者使光标到行首，后者使光标下移一格。
            for text in content:
                # 分词
                word_list = list(jieba.cut(text, cut_all=False))
                # 去停用词
                word_list = self.rm_stop_words(word_list)
                
                dictionary.append(' '.join(word_list))

        for i in range(dictionary.__len__()):
            file = self.root + 'rent_corpus.txt'
            f = open(file, 'a+', encoding='utf-8')
            f.write(''.join(dictionary[i]) +'\n')   #加\n换行显示

    #  训练生成词向量
    def model_train(self, train_file_name, save_model_name):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.Text8Corpus(train_file_name, max_sentence_length = 1000)  # 加载语料
        model = gensim.models.Word2Vec(sentences, size=128, window=4, sg = 1, hs = 0,
                                negative = 10, alpha=0.03, min_alpha=0.0007, seed = 14, min_count=20)
        # 训练skip-gram模型; window=4
        model.save(save_model_name)
        model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)   # 以二进制类型保存模型以便重用
    
    def trainVec(self):
        if not os.path.exists(self.save_model_name):     # 判断文件是否存在
            self.model_train(self.train_file, self.save_model_name)
        else:
            print('此训练模型已经存在，不用再次训练')

    #  生成扩展词
    def generate_expan(self, query):
        if query in self.vocab:     #（要在词典内）
            # 计算扩展词（相关词）列表
            expansion = self.word_model.most_similar(query, topn=10)  # 10个最相关的
        else:
            expansion = [("", 0)]
        return expansion

if __name__ == '__main__':
    start = datetime.now()
    test = queryExpan()
#    test.generate_corpors()
#    test.trainVec()
    #    query = '房贷'
    query = sys.argv[1]
    result = test.generate_expan(query)

    print('input query:%s' % query)
    for expansion, relation in result:
        print(u'query扩展：',query + expansion, relation)
    print("-------------------------------\n")
    print("This took ", datetime.now() - start)

