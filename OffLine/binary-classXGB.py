#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

class binaryXGB(object):
    def __init__(self):
        self.root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

    def prepared(self):
        trainDf=pd.read_csv(self.root+'trainset_CF.csv')
        trainDf.head()

        trainDf = trainDf.sample(6000)
        print ('dataset shape is', file=sys.stderr)
        print(trainDf.shape)

        X = trainDf[['userCFScore', 'itemCFScore', 'popular']]  #选择表格中的'w'、'z'列
        y = trainDf.label

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        return X_train, X_test, y_train, y_test

    ############################Modeling########################
    def modeling(self, X_train, X_test, y_train, y_test):
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, learning_rate =0.13, n_estimators=100,
                                      subsample=0.8, max_depth=3, min_child_weight=5, gamma=0, colsample_bytree=0.7,
                                      reg_lambda=1e-5, eval_metric="auc")

        xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
        print ('model training succ', file=sys.stderr)

        y_pred = xgb_model.predict(X_test)
        print('model test accuracy', accuracy_score(y_test, y_pred))
    ############################推荐结果输出######################
    def recommend(self):
        print ('reading testdata and predicting ctr', file=sys.stderr)
        
        testDf = pd.read_csv(self.root+"testset_CF.csv")
        X_test = np.matrix( pd.DataFrame(testDf, index=None, columns=['userCFScore', 'itemCFScore', 'popular']) )
        y_test = np.array(testDf.label)
        user_id = testDf.user_id
        new_id = testDf.new_id

        fout = open(root+"result_Xgb.csv", 'w')
        fout.write(",".join(["user_id", "new_id", "pred_label", "0_proba", "1_proba"]) + "\n")

        nrows = len(X_test)                   #测试集样本量
        Xp = np.matrix(X_test)                #测试集特征矩阵
        yp = np.zeros((nrows, 3))             #结果及分数
        for i in range(0, nrows):
            xp = Xp[i, :]                                   #取第i个样本
            yp[i, 0] = xgb_model.predict(xp)                #预测结果
            yp[i, 1] = xgb_model.predict_proba(xp)[0][0]    #预测为0的概率
            yp[i, 2] = xgb_model.predict_proba(xp)[0][1]    #预测为1的概率
            
            fout.write(",".join( map( lambda x: str(x), [user_id[i], new_id[i], yp[i, 0], yp[i, 1], yp[i, 2]] ) ) + "\n")

        fout.close()

if __name__ == '__main__':
    start = datetime.now()
    
    test = binaryXGB()
    X_train, X_test, y_train, y_test = test.prepared()
    test.modeling(X_train, X_test, y_train, y_test)
#    test.recommend()

    print("This took ", datetime.now() - start)

