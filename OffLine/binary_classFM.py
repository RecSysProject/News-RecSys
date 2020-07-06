#!/usr/bin/env python
from __future__ import division

import pandas as pd
import numpy as np
from math import exp
from datetime import datetime
from random import normalvariate  # 正态分布
from sklearn.model_selection import train_test_split

root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

trainDf=pd.read_csv(root+'trainset_CF.csv')

X = trainDf[['userCFScore', 'itemCFScore', 'popular']]  #选择表格中的'w'、'z'列
y = trainDf.label

print('dataset festure shape is', X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.001, random_state=42)

y_train = y_train.map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
y_test = y_test.map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

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
w_0, w, v = SGD_FM(np.mat(X_train), y_train, 20, 100)
print(
      "训练准确性为：%f" % (1 - getAccuracy(np.mat(X_train), y_train, w_0, w, v)))

Train_end = datetime.now()
print("训练用时为：%s" % (Train_end - Train_start))

print("开始测试")
print("测试准确性为：%f" % (1 - getAccuracy(np.mat(X_test), y_test, w_0, w, v)))
