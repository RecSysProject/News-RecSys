#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys
import warnings
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

#导入数据
root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'

trainDf=pd.read_csv(root+'trainset_CF.csv')
trainDf.head()

trainDf = trainDf.sample(3000)
print ('training data shape is', file=sys.stderr)
print(trainDf.shape)

X = trainDf[['userCFScore', 'itemCFScore', 'popular']]  #选择表格中的'w'、'z'列

y = trainDf.label

X = np.array(X)
y = np.array(y)

#Modeling
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc", max_depth=3,
                              min_child_weight=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
print ('model training succ', file=sys.stderr)

#推荐结果输出

print ('reading testdata and predicting ctr', file=sys.stderr)

testDf = pd.read_csv(root+"testset_CF.csv")


X_test = np.matrix( pd.DataFrame(testDf, index=None, columns=['userCFScore', 'itemCFScore', 'popular']) )
y_test = np.array(testDf.label)
user_id = testDf.user_id
new_id = testDf.new_id

fout = open(root+"result_Xgb.csv", 'w')
fout.write(",".join(["user_id", "new_id", "pred_label", "0_proba", "1_proba"]) + "\n")

nrows = len(X_test)                   #测试集样本量
Xp = np.matrix(X_test)               #测试集特征矩阵
yp = np.zeros((nrows, 3))            #结果及分数
for i in range(0, nrows):
    xp = Xp[i, :]                                   #取第i个样本
    yp[i, 0] = xgb_model.predict(xp)                  #预测结果
    yp[i, 1] = xgb_model.predict_proba(xp)[0][0]    #预测为0的概率
    yp[i, 2] = xgb_model.predict_proba(xp)[0][1]    #预测为1的概率
    
    fout.write(",".join( map( lambda x: str(x), [user_id[i], new_id[i], yp[i, 0], yp[i, 1], yp[i, 2]] ) ) + "\n")

fout.close()

