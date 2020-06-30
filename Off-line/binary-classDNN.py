#!/usr/bin/env python

#Data prepared
#user、news特征
import numpy as np
import _pickle as cPickle
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

batch_size = 128
epochs = 5

num_classes = 2   #新闻id个数
length = 384     #输入用户向量


root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'
user_feature = cPickle.load( open(root + 'binary_user_feature.pkl','rb') )   #正负样本
news_feature = cPickle.load( open(root + 'binary_news_feature.pkl','rb') )  #新闻embedding集合
target = cPickle.load( open(root + 'binary_label.pkl','rb') )              #用户画像矩阵

#归一化user、news特征
news_feature_scaled = preprocessing.scale(news_feature)
user_feature_scaled = preprocessing.scale(user_feature)

#合并特征
feature = np.hstack((news_feature_scaled, user_feature_scaled))

#划分训练集与测试集

X_train, X_test, y_train, y_test = train_test_split(feature, target, train_size=0.7, random_state=42)
X1_train = X_train[:, :384]  #取前384列
X2_train = X_train[:, 384:]  #取后128列
X1_test = X_test[:, :384]  #取前384列
X2_test = X_test[:, 384:]  #取后128列

print('训练样本1的维度:', X1_train.shape)
print(X1_train.shape[0], '个训练样本1')
print('训练样本2的维度', X2_train.shape)
print(X2_train.shape[0], '个训练样本2')

print('测试样本1的维度:', X1_test.shape)
print(X1_test.shape[0], '个测试样本1')
print('测试样本2的维度', X2_test.shape)
print(X2_test.shape[0], '个测试样本2')

#binary-classDNN
#target Onehot

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

encoded_y_train = encode(y_train)
encoded_y_test = encode(y_test)

import sys
import keras as K
import tensorflow as tf
from keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import TensorBoard

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

K.backend.clear_session()

print("Using Python version " + str(py_ver))
print("Using Keras version " + str(k_ver))
print("Using TensorFlow version " + str(tf_ver))

#双输入
"定义第一个输入层，user_feature"
input1 = K.layers.Input(shape=(384,))  #(6464, 384)

"降维层"
reduce = K.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input1) # 用户向量降维到128

'BN输出层'
output1 = K.layers.BatchNormalization()(reduce)

"定义第二个输入层，news_feature"
input2 = K.layers.Input(shape=(128,))  #(6464, 128)

"定义第二个输出层"
output2 = K.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input2)

'点积层'
# doted = K.layers.Dot(1)([reduce,output2])
doted = K.layers.Dot(1)([output1,output2])

'最后输出层'
output3 = K.layers.Dense(2, activation='softmax', kernel_regularizer=l2(1e-4))(doted)

merge_model = K.models.Model(inputs=[input1, input2], outputs=output3)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
merge_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

"输出模型各层的参数"
merge_model.summary()

# TensorBoard调用查看一下训练情况
#tb_cb = TensorBoard(log_dir='logs')

print("Starting training ")
#
#h = merge_model.fit([X1_train, X2_train], encoded_y_train, batch_size=batch_size, epochs=10, shuffle=True, verbose=1,
#                    callbacks=[tb_cb])
h = merge_model.fit([X1_train, X2_train], encoded_y_train, batch_size=batch_size, epochs=10, shuffle=True, verbose=1)

print("Training finished \n")

#评估模型
eval = merge_model.evaluate([X1_test, X2_test], encoded_y_test, verbose=0)
print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
      % (eval[0], eval[1] * 100) )

#plot_model(model=merge_model, to_file='binary-class_merge_DNN.png', show_shapes=True)
