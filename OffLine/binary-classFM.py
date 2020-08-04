#!/usr/bin/env python
from __future__ import division

import pandas as pd
import numpy as np
from math import exp
from datetime import datetime
from random import normalvariate  # 正态分布
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

trainDf=pd.read_csv(root+'trainset_CF.csv')
trainDf = trainDf.sample(6000)

X = trainDf[['userCFScore', 'itemCFScore', 'popular']]  #选择表格中的'w'、'z'列
y = trainDf.label

#特征分箱
##等频分箱
def encode(data):
    data = np.array(data).reshape([-1, 1])
    Encoder = OneHotEncoder()
    Encoder.fit(data)
    encoded = Encoder.transform(data).toarray()
    return encoded

def bin_frequency(x,y,n=10): # x为待分箱的变量，y为target变量.n为分箱数量
    total = y.count()  # 计算总样本数
    bad = y.sum()      # 计算坏样本数
    good = y.count()-y.sum()  # 计算好样本数
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.qcut(x,n)})  # 用pd.cut实现等频分箱
    d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin'])
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count() # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
    cut = []
    cut.append(float('-inf'))
    for i in d4.min_bin:
        cut.append(i)
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe

d4,iv,cut,woe = bin_frequency(X['itemCFScore'], y, n =4)
temp = pd.cut(X['itemCFScore'], cut, labels=False)
X_item = encode(temp)

d4,iv,cut,woe = bin_frequency(X['userCFScore'], y, n =2)
temp = pd.cut(X['userCFScore'], cut, labels=False)
X_user = encode(temp)

d4,iv,cut,woe = bin_frequency(X['popular'], y, n =5)
temp = pd.cut(X['popular'], cut, labels=False)
X_popular = encode(temp)

temp = np.hstack((X_user, X_item))
X_discretization = np.hstack((temp, X_popular))
#print('离散化后的feature shape', X_discretization.shape)

X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(X_discretization, y, train_size=0.6, random_state=42)

y_train_freq = y_train_freq.map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
y_test_freq = y_test_freq.map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1

X_train_freq = np.array(X_train_freq)
X_test_freq = np.array(X_test_freq)
y_train_freq = np.array(y_train_freq)
y_test_freq = np.array(y_test_freq)

print('trainset_freq feature shape is', X_train_freq.shape)
print('testset_freq feature shape is', X_test_freq.shape)

# GBDT分箱
gbc = GradientBoostingClassifier(n_estimators=2, learning_rate=0.12,
                                 max_depth=3, subsample=0.83)
gbc.fit(X, y)

one_hot = OneHotEncoder()
X_gb = one_hot.fit_transform(gbc.apply(X)[:, :, 0])
X_gb = X_gb.todense()

X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(X_gb, y, train_size=0.6, random_state=42)

y_train_gb = y_train_gb.map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
y_test_gb = y_test_gb.map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1

y_train_gb = np.array(y_train_gb)
y_test_gb = np.array(y_test_gb)

print('trainset_gb feature shape is', X_train_gb.shape)
print('testset_gb feature shape is', X_test_gb.shape)

# FM——modeling

def sigmoid(inx):
    return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
#     return 1.0 / (1 + exp(-inx))

def SGD_FM(dataMatrix, classLabels, k, iter):
    '''
        :param dataMatrix:  特征矩阵
        :param classLabels: 类别矩阵
        :param k:           辅助向量的大小
        :param iter:        迭代次数
        :return:
        '''
    # dataMatrix用的是mat, classLabels是列表
    m, n = np.shape(dataMatrix)   #矩阵的行列数，即样本数m和特征数n
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = np.zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    v = normalvariate(0, 0.2) * np.ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数
    
    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  #二阶交叉项的计算
            interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.       #二阶交叉项计算完成
            
            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            loss = 1-sigmoid(classLabels[x] * p[0, 0])    #计算损失
            
            w_0 = w_0 +alpha * loss * classLabels[x]
            
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * (
                                                                            dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
    if not it%10:
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v

def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = np.shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
        
        pre = sigmoid(p[0, 0])
        result.append(pre)
        
        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem

print("开始训练")
Train_start = datetime.now()
w_0, w, v = SGD_FM(np.mat(X_train_gb), y_train_gb, 20, 60)
print(
      "训练准确性为：%f" % (1 - getAccuracy(np.mat(X_train_gb), y_train_gb, w_0, w, v)))

Train_end = datetime.now()
print("训练用时为：%s" % (Train_end - Train_start))

print("开始测试")
print("测试准确性为：%f" % (1 - getAccuracy(np.mat(X_test_gb), y_test_gb, w_0, w, v)))
