#!/usr/bin/env python

import sys
from operator import itemgetter
import math
import random
random.seed(0)

trainset = {}  #
testset = {}

n_sim_news= 200      #相似新闻数量       *重要5，10，20，40，(80,0.1406)最好，100
n_rec_news = 10     #推荐电影数量

news_sim_mat = {}    #item间的相似度矩阵
news_popular = {}   #item ：流行度
news_count = 0

print ('Similar user number = %d' % n_sim_news, file=sys.stderr)
print ('recommended movie number = %d' % n_rec_news, file=sys.stderr)

# global_param
dataset='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/data/train_date_set1.txt'
root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

# f1=open(root+'data/train_date_set1/train_date_set1_1.txt')
f1=open(root+'data/train_date_set1.txt')

#划分数据集

pivot = 0.6

trainset_len = 0
testset_len = 0

"遍历所有浏览记录，保存所有用户浏览过的新闻id到字典，返回{用户id：news_id1, news_id2…}"

for line in f1.readlines():
    user = line.strip().split('\t')[0]
    news = line.strip().split('\t')[1]
    time = line.strip().split('\t')[2]
    
    if random.random() < pivot:
        trainset.setdefault(user, {})
        trainset[user][news] = time
        trainset_len += 1
    else:
        testset.setdefault(user, {})
        testset[user][news] = time
        testset_len += 1

print ('testset_len %s generate' % testset_len, file=sys.stderr)
print ('trainset_len %s generate' % trainset_len, file=sys.stderr)

print ('the number of user %s trainset' % len(trainset), file=sys.stderr) #用户数是一样的
print ('the number of user %s testset' % len(testset), file=sys.stderr) #用户数是一样的

#计算item总数及热度
for user, news in trainset.items():
    for new in news:
        # 统计 item 流行度
        if new not in news_popular:
            news_popular[new] = 0
        news_popular[new] += 1

print('count news number and popularity succ', file=sys.stderr)

# save the total number of movies
news_count = len(news_popular)
print('total news number = %d' % news_count, file=sys.stderr)

#计算同时看过两个item的用户数
from collections import defaultdict

itemsim_mat = news_sim_mat       # item，相关item， 同时看过它俩的用户数

#同时看过这两个不同电影的用户数
for user, movies in trainset.items():
    for m1 in movies:
        itemsim_mat.setdefault(m1, defaultdict(int))
        for m2 in movies:
            if m1 == m2:
                continue
            itemsim_mat[m1][m2] += 1

#计算item相似度矩阵
#把同时看过两个item的用户数转化为相似度
#同时看过m1、m2新闻的用户数/根号下（看过新闻m1的用户数*看过新闻m2的用户数）
#itemsim_mat item，相关item， 同时看过它俩的用户数

simfactor_count = 0
PRINT_STEP = 200000

for m1, related_news in itemsim_mat.items():
    for m2, count in related_news.items():
        itemsim_mat[m1][m2] = count / math.sqrt(news_popular[m1] * news_popular[m2])
        simfactor_count += 1
        if simfactor_count % PRINT_STEP == 0:
            print('calculating movie similarity factor(%d)' % simfactor_count, file=sys.stderr)

print('calculate movie similarity matrix(similarity factor) succ', file=sys.stderr)
print('Total similarity factor number = %d' % simfactor_count, file=sys.stderr)

#推荐
# itemsim_mat     电影m1：相关电影m2：       相似度
# trainset            用户u ：用户已看过的item： 评分

def recommend(user):
    
    K = n_sim_news         #取相似度最高的k个item
    N = n_rec_news         #推荐前N个item
    rank = {}
    
    watched_news = trainset[user]   # 用户已看过的电影
    
    "遍历该用户看过的所有item"
    for new, time in watched_news.items():
        
        '遍历与这个看过的item相似度最高的前k个'
        for related_news, similarity_factor in sorted(news_sim_mat[new].items(),
                                                      key=itemgetter(1), reverse=True)[:K]:
            if related_news in watched_news:  #去除用户已看过的
                continue
            
            "预测该用户对每个候选item的兴趣"
            rank.setdefault(related_news, 0)
            
            "只要与该用户看过的item相似，就将这个item与用户看过的item的相似度累加对到这个item的兴趣上"
            rank[related_news] += similarity_factor

    "推荐兴趣最高的前N个item"
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

#评估
''' print evaluation result: precision, recall, coverage and popularity '''
print ('Evaluation start...', file=sys.stderr)

N = n_rec_news
# precision and recall 准确率和召回率参数
hit = 0
rec_count = 0
test_count = 0
# coverage覆盖率参数
all_rec_news = set()
# popularity流行度参数
popular_sum = 0

for i, user in enumerate(trainset):
    if i % 500 == 0:
        print ('recommended for %d users' % i, file=sys.stderr)
    "测试集中该用户读过的news"
    test_news = testset.get(user, {})
    "推荐给该用户的news"
    rec_news = recommend(user)

    for news, _ in rec_news:
        "如果推荐给该用户的news在测试集中用户确实读过，命中数加一"
        if news in test_news:
            hit += 1
        "加入所有推荐过的新闻集合"
        all_rec_news.add(news)
        
        popular_sum += math.log(1 + news_popular[news])     #总{ (推荐的新闻热度+1)取对数 }
rec_count += N                            #总推荐过的新闻数
test_count += len(test_news)         #总用户确实读过的新闻数

precision = hit / (1.0 * rec_count)       #准确率=命中数/总推荐过的新闻数
recall = hit / (1.0 * test_count)           #召回率=命中数/总用户读过的新闻数，测试集中
coverage = len(all_rec_news) / (1.0 * news_count)      #覆盖率=推荐过的新闻集合/总新闻数
popularity = popular_sum / (1.0 * rec_count)            #流行度=推荐过的新闻流行度总和/推荐过的新闻数累加

print ('K=%d precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
       (n_sim_news, precision, recall, coverage, popularity), file=sys.stderr)



