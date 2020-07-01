#!/usr/bin/env python
import sys
import math
import pandas as pd
from operator import itemgetter
from collections import defaultdict

import random
random.seed(0)

############################划分正负样本#################################
dataset='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/data/train_date_set1.txt'
root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

f1=open(root+'data/train_date_set1.txt')

#positive
posSet = {}   #正样本
dataSet = {}  #总体样本

label = 1
posSet_len = 0
dataSet_len = 0

"遍历所有浏览记录，保存所有用户浏览过的新闻id到字典，返回{用户id：news_id1, news_id2…}"

for line in f1.readlines():
    user = line.strip().split('\t')[0]
    news = line.strip().split('\t')[1]
    time = line.strip().split('\t')[2]
    title = line.strip().split('\t')[3]
    
    posSet.setdefault(user, {})
    dataSet.setdefault(user, {})
    
    posSet[user][news] = label
    dataSet[user][news] = label
    posSet_len += 1
    dataSet_len += 1

print ('posSet_len %s generate' % posSet_len, file=sys.stderr)

news_popular = {}   #item ：流行度
news_count = 0

for user, news in posSet.items():
    for new in news:
        # 统计 item 流行度
        if new not in news_popular:
            news_popular[new] = 0
        news_popular[new] += 1

print('count news number and popularity succ', file=sys.stderr)

# save the total number of movies
news_count = len(news_popular)
print('total news number = %d' % news_count, file=sys.stderr)

#negative
#保证正负样本均衡1:1
#取用户没有阅读过的，热度比较高的news为负样本
negSet = {}

label = 0
negSet_len = 0

for user, news in posSet.items():
    watched_news = posSet[user]
    postive_len = len(news)
    for news, popular in sorted(news_popular.items(), key=itemgetter(1), reverse=True)[0:2*postive_len]:
        if news in watched_news:  #去除用户已看过的
            continue
        if postive_len > 0 :
            negSet.setdefault(user, {})
            negSet[user][news] = label
            dataSet[user][news] = label
            
            postive_len -= 1
            negSet_len += 1
            dataSet_len += 1

# dataSet_len = posSet_len+negSet_len

print ('posSet_len %s generate' % posSet_len, file=sys.stderr)
print ('negSet_len %s generate' % negSet_len, file=sys.stderr)
print ('dataSet_len %s generate' % dataSet_len, file=sys.stderr)

#创建文件
f=open(root+'dataset_CF.csv','w')
ocolnames = ['user_id', 'new_id', 'label']
f.write( ','.join(ocolnames) + '\n' )

for user, news in dataSet.items():
    for new, label in news.items():
        f.write( str(user) +','+ str(new) + ',' + str(label) + '\n')
f.close()

#############################提取特征#################################
#userCF
#item-users倒排表
news2users = dict()

for user, news in posSet.items():
    for new in news:
        #  item-users 倒排表
        if new not in news2users:
            news2users[new] = set()
        news2users[new].add(user)

#用户与相关用户共同看过的news数
user_sim_mat = {}    #用户间的相似度矩阵
usersim_mat = user_sim_mat       # 用户，相关用户， 共同看过的电影数

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
        usersim_mat[u][v] = count / math.sqrt(len(posSet[u]) * len(posSet[v]))
        simfactor_count += 1
        if simfactor_count % PRINT_STEP == 0:
            print ('calculating user similarity factor(%d)' % simfactor_count, file=sys.stderr)


#计算正负样本对用户的推荐度
# usersim_mat 用户u ：相关用户v ：         相似度
# trainset       用户u ：用户已看过的item： 评分
n_sim_user = 20

def userCFScore(user):
    
    K = n_sim_user    #取相似度最高的k个用户
    
    rank = dict()
    sample_news = posSet[user].copy()    # 用户已看过的电影
    negtive_news = negSet[user].copy()      # 用户没看过的热门电影
    sample_news.update(negtive_news)    #合并
    "遍历与该用户相似度最高的k个用户，用户、相似度"
    for similar_user, similarity_factor in sorted(user_sim_mat[user].items(),
                                                  key=itemgetter(1), reverse=True)[0:K]:
        "遍历这些相似用户看过的所有item"
        for new in posSet[similar_user]:
            
            "预测该用户对每个正负item的兴趣"
            if new in sample_news:
                rank.setdefault(new, 0)
                
                "只要相关用户看过这个item，就将该用户与这个相关用户的相似度累加对到这个item的兴趣上"
                rank[new] += similarity_factor

    "返回用户正负样本及对应的推荐度"
    return rank      #推荐兴趣最高的前N个item

#插入userCFSocre特征
#生成特征
for user, news in dataSet.items():
    rec_news = userCFScore(user)
    for new in news:
        if new in rec_news:
            dataSet[user][new] = rec_news[new]

#插入userCFScore

userCFScore = []

for user, news in dataSet.items():
    for new, Score in news.items():
        userCFScore.append(Score)

userCFScore = pd.DataFrame(userCFScore, columns=['userCFScore'])

df=pd.read_csv(root+'dataset_CF.csv')

df['userCFScore'] = userCFScore #将新列的名字设置为userCFScore
df.to_csv(root+'dataset_CF.csv', index = None)

#itemCF
#计算同时看过两个item的用户数
from collections import defaultdict
news_sim_mat ={}

itemsim_mat = news_sim_mat       # item，相关item， 同时看过它俩的用户数

#同时看过这两个不同电影的用户数
for user, movies in posSet.items():
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

print('calculate movie similarity matrix(similarity factor) succ',
      file=sys.stderr)
print('Total similarity factor number = %d' %
      simfactor_count, file=sys.stderr)

#计算正负样本对用户的推荐度
# itemsim_mat     电影m1：相关电影m2：       相似度
# trainset            用户u ：用户已看过的item： 评分

n_sim_news = 20

def itemCFScore(user):
    
    K = n_sim_news         #取相似度最高的k个item
    
    rank = {}
    
    watched_news = posSet[user]   # 用户已看过的电影
    sample_news = negSet[user]      # 用户没看过的热门电影
    sample_news.update(watched_news)    #合并
    
    "遍历该用户看过的所有item"
    for new, time in watched_news.items():
        
        '遍历与这个看过的item相似度最高的前k个'
        for related_news, similarity_factor in sorted(news_sim_mat[new].items(),
                                                      key=itemgetter(1), reverse=True)[:K]:
            if related_news in sample_news:
                
                "预测该用户对每个正负item的兴趣"
                rank.setdefault(related_news, 0)
                
                "只要与该用户看过的item相似，就将这个item与用户看过的item的相似度累加对到这个item的兴趣上"
                rank[related_news] += similarity_factor

    "返回用户正负样本及对应的推荐度"
    return rank

#插入itemCFScore特征
#生成特征
for user, news in dataSet.items():
    rec_news = itemCFScore(user)
    for new in news:
        if new in rec_news:
            dataSet[user][new] = rec_news[new]

#插入itemCF

itemCFScore = []

for user, news in dataSet.items():
    for new, Score in news.items():
        itemCFScore.append(Score)

itemCFScore = pd.DataFrame(itemCFScore, columns=['itemCFScore'])

df=pd.read_csv(root+'dataset_CF.csv')

df['itemCFScore'] = itemCFScore #将新列的名字设置为itemCFScore
df.to_csv(root+'dataset_CF.csv', index = None)

#popular
#生成特征
for user, news in dataSet.items():
    for new in news:
        dataSet[user][new] =math.log(1 + news_popular[new])

#插入popular
popular = []

for user, news in dataSet.items():
    for new, Score in news.items():
        popular.append(Score)

popular = pd.DataFrame(popular, columns=['popular'])

df=pd.read_csv(root+'dataset_CF.csv')

df['popular'] = popular #将新列的名字设置为itemCFScore
df.to_csv(root+'dataset_CF.csv', index = None)

#############################划分训练集与测试集#################################
pivot=0.7

trainset_len = 0
testset_len = 0

f2=open(root+'dataset_CF.csv')
f3 = open(root+'trainset_CF.csv','w+')
f3.write(",".join(["user_id", "new_id", "label", "userCFScore", "itemCFScore", "popular"]) + "\n")

f4 = open(root+'testset_CF.csv','w+')

for line in f2.readlines():
    # split the data by pivot
    if random.random() < pivot:
        
        f3.write(line)
        trainset_len += 1
    else:
        f4.write(line)
        testset_len += 1


f2.close()
f3.close()
f4.close()

print ('split training set and test set succ', file=sys.stderr)
print ('train set = %s' % trainset_len, file=sys.stderr)
print ('test set = %s' % testset_len, file=sys.stderr)


