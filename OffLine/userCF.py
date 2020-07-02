#!/usr/bin/env python

import sys
import math
from datetime import datetime
from operator import itemgetter
from collections import defaultdict
import random
random.seed(0)

class userCF(object):
    def __init__(self):
        self.trainset = {}  #
        self.testset = {}

        self.n_sim_user = 80      #相似用户数量       *重要5，10，20，40，(80,0.1406)最好，100
        self.n_rec_news = 10      #推荐news数量

        self.user_sim_mat = {}    #用户间的item相似度矩阵
        self.news_popular = {}    #item ：流行度
        self.news_count = 0
        self.root = '/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

#        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
#        print ('recommended news number = %d' % self.n_rec_news, file=sys.stderr)

    #划分数据集
    def split_dataset(self):
        # global_param
        f1=open(self.root+'data/train_date_set1.txt')

        pivot = 0.6
        trainset_len = 0
        testset_len = 0

        "遍历所有浏览记录，保存所有用户浏览过的新闻id到字典，返回{用户id：news_id1, news_id2…}"
        for line in f1.readlines():
            user = line.strip().split('\t')[0]
            news = line.strip().split('\t')[1]
            time = line.strip().split('\t')[2]
            
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][news] = time
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][news] = time
                testset_len += 1

#        print ('testset_len %s generate' % testset_len, file=sys.stderr)
#        print ('trainset_len %s generate' % trainset_len, file=sys.stderr)
#
#        print ('the number of user %s trainset' % len(self.trainset), file=sys.stderr) #用户数是一样的
#        print ('the number of user %s testset' % len(self.testset), file=sys.stderr) #用户数是一样的

    def user_sim(self):
        #item-users倒排表
        news2users = dict()

        for user, news in self.trainset.items():
            for new in news:
                #  item-users 倒排表
                if new not in news2users:
                    news2users[new] = set()
                news2users[new].add(user)
                # 同时统计 item 流行度
                if new not in self.news_popular:
                    self.news_popular[new] = 0
                self.news_popular[new] += 1

        self.news_count = len(news2users)  ## item 总数 2194 保存用于评估

        #用户与相关用户共同看过的news数
        usersim_mat = self.user_sim_mat       # 用户，相关用户， 共同看过的电影数

        #两个不同用户看过的相同电影的个数
        for news, users in news2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1

        #用户相似度矩阵
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
#                if simfactor_count % PRINT_STEP == 0:
#                    print ('calculating user similarity factor(%d)' % simfactor_count, file=sys.stderr)

    #推荐
    # usersim_mat    用户u ：相关用户v ：         相似度
    # trainset       用户u ：用户已看过的item：    评分
    def recommend(self, user):
        
        K = self.n_sim_user   #取相似度最高的k个用户
        N = self.n_rec_news   #推荐前N个item
        rank = dict()
        watched_news = self.trainset[user]    # 用户已看过的电影
        
        "遍历与该用户相似度最高的k个用户，用户、相似度"
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            "遍历这些相似用户看过的所有item"
            for new in self.trainset[similar_user]:
                if new in watched_news:         #去掉已看过的
                    continue
            
                "预测该用户对每个候选item的兴趣"
                rank.setdefault(new, 0)
                
                "只要相关用户看过这个item，就将该用户与这个相关用户的相似度累加对到这个item的兴趣上"
                rank[new] += similarity_factor

        "推荐兴趣最高的前N个item"
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]    #推荐兴趣最高的前N个item

    #评估
    ''' print evaluation result: precision, recall, coverage and popularity '''
    def evaluation(self):
        
        N = self.n_rec_news
        # precision and recall 准确率和召回率参数
        hit = 0
        rec_count = 0
        test_count = 0
        # coverage覆盖率参数
        all_rec_news = set()
        # popularity流行度参数
        popular_sum = 0
        
        print ('Evaluation start...', file=sys.stderr)
        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            "测试集中该用户读过的news"
            test_news = self.testset.get(user, {})
            "推荐给该用户的news"
            rec_news = self.recommend(user)

            for news, _ in rec_news:
                "如果推荐给该用户的news在测试集中用户确实读过，命中数加一"
                if news in test_news:
                    hit += 1
                "加入所有推荐过的新闻集合"
                all_rec_news.add(news)
                
                popular_sum += math.log(1 + self.news_popular[news])     #总{ (推荐的新闻热度+1)取对数 }
            rec_count += N                            #总推荐过的新闻数
            test_count += len(test_news)              #总用户确实读过的新闻数

        precision = hit / (1.0 * rec_count)       #准确率=命中数/总推荐过的新闻数
        recall = hit / (1.0 * test_count)         #召回率=命中数/总用户读过的新闻数，测试集中
        coverage = len(all_rec_news) / (1.0 * self.news_count)       #覆盖率=推荐过的新闻集合/总新闻数
        popularity = popular_sum / (1.0 * rec_count)            #流行度=推荐过的新闻流行度总和/推荐过的新闻数累加
        
        print ('K=%d precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' % (self.n_sim_user, precision, recall, coverage, popularity), file=sys.stderr)

if __name__ == '__main__':
    start = datetime.now()
    test = userCF()
    test.split_dataset()
    test.user_sim()
    test.evaluation()
    result = test.recommend(sys.argv[1])
    for item_id, ctr in result:
        print('item_id:%s, ctr:%.4f' % (item_id, ctr))
    print("This took ", datetime.now() - start)
