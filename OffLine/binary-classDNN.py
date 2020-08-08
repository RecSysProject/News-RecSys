#!/usr/bin/env python

import numpy as np
import _pickle as cPickle
from datetime import datetime
from operator import itemgetter
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import sys
import keras as K
import tensorflow as tf
from keras.utils import plot_model
from keras.models import load_model
from keras.regularizers import l2
from keras.callbacks import TensorBoard

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

K.backend.clear_session()

print("Using Python version " + str(py_ver))
print("Using Keras version " + str(k_ver))
print("Using TensorFlow version " + str(tf_ver))

class binaryDNN(object):
    def __init__(self):
        self.root='/Users/tung/Python/PersonalProject/NewsRecommend/Off-line/'
        '训练参数'
        self.batch_size = 128
        self.epochs = 10
        self.num_classes = 2      #label个数
        self.user_len = 512       #输入用户向量长度
        self.news_len = 256       #输入新闻向量长度
        'recommend'
        self.N = 10               #推荐前N个item
        self.user_embedding = {}
        self.news_embedding = {}
        self.news_id = []
    
#####################################binary-classDNN###########################################
    def modeling(self):
        ###################################Data prepared#######################################
        '词嵌入'
        word_user_feature = np.array(cPickle.load( open(self.root + 'binary_user_word.pkl','rb')))  #用户序列内容嵌入
        word_news_feature = np.array(cPickle.load( open(self.root + 'binary_news_word.pkl','rb')))  #新闻词嵌入
        target = np.array(cPickle.load( open(self.root + 'binary_label.pkl','rb')))                 #label
        'deepWalk嵌入'
        graph_user_feature = np.array(cPickle.load( open(self.root + 'binary_user_graph.pkl','rb')))    #用户序列图嵌入
        graph_news_feature = np.array(cPickle.load( open(self.root + 'binary_news_graph.pkl','rb')))    #新闻图嵌入
        '词、图嵌入合并'
        user_feature = np.hstack((word_user_feature, graph_user_feature))    #384+128
        news_feature = np.hstack((word_news_feature, graph_news_feature))    #128+128
        '归一化user、news特征'
        news_feature_scaled = preprocessing.scale(news_feature)
        user_feature_scaled = preprocessing.scale(user_feature)
        '合并特征'
        feature = np.hstack((news_feature_scaled, user_feature_scaled))
        
        'split_dataset'
        X_train, X_test, y_train, y_test = train_test_split(feature, target, train_size=0.7, random_state=42)
        X1_train = X_train[:, :self.user_len]  #取前512列
        X2_train = X_train[:, self.user_len:]  #取后256列
        X1_test = X_test[:, :self.user_len]  #取前512列
        X2_test = X_test[:, self.user_len:]  #取后256列
        
        print('训练样本1的维度:', X1_train.shape)
        print(X1_train.shape[0], '个训练样本1')
        print('训练样本2的维度', X2_train.shape)
        print(X2_train.shape[0], '个训练样本2')
        
        print('测试样本1的维度:', X1_test.shape)
        print(X1_test.shape[0], '个测试样本1')
        print('测试样本2的维度', X2_test.shape)
        print(X2_test.shape[0], '个测试样本2')
        
        'target Onehot'
        encoded_y_train = to_categorical(y_train)
        encoded_y_test = to_categorical(y_test)
        
        'Define model structure'
        '定义第一个输入层，user_feature'
        input1 = K.layers.Input(shape=(self.user_len,))  #(6464, 512)
        "降维层"
        reduce1 = K.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input1) # 用户向量降维到128
        "BN输出层"
        output1 = K.layers.BatchNormalization()(reduce1)

        '定义第二个输入层，news_feature'
        input2 = K.layers.Input(shape=(self.news_len,))  #(6464, 256)
        "降维层"
        reduce2 = K.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(input2)
        "BN输出层"
        output2 = K.layers.BatchNormalization()(reduce2)

        '点积层'
        doted = K.layers.Dot(1)([output1,output2])

        '最后输出层'
        output3 = K.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(1e-4))(doted)
        merge_model = K.models.Model(inputs=[input1, input2], outputs=output3)
        merge_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        '输出模型各层的参数'
        merge_model.summary()

        tb_cb = TensorBoard(log_dir='logs')
        print("Starting training ")

        h = merge_model.fit([X1_train, X2_train], encoded_y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=True, verbose=1,callbacks=[tb_cb])

        print("Training finished \n")

        'Evaluation'
        eval = merge_model.evaluate([X1_test, X2_test], encoded_y_test, verbose=0)
        print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100) )
        plot_model(model=merge_model, to_file=self.root + 'models/binary-class_merge_DNN.png', show_shapes=True)

        'save model'
        model_path = self.root + 'models/Dual_input_biinaryNN.h5'
        merge_model.save(model_path)
        del merge_model

    def prepared(self):
        ######################recommend####################
        '词嵌入'
        word_user_feature = cPickle.load( open(self.root + 'multi_user_word.pkl','rb') )     #用户画像、时间上下文
        word_news_feature = cPickle.load( open(self.root + 'multi_news_word.pkl','rb') )     #新闻矩阵
        'deepWalk嵌入'
        graph_user_feature = cPickle.load( open(self.root + 'multi_user_graph.pkl','rb') )   #正负样本
        graph_news_feature = cPickle.load( open(self.root + 'multi_news_graph.pkl','rb') )
    
        'user特征'
        temp_user = [word_user_feature,  graph_user_feature]
        for _ in temp_user:
            for k, v in _.items():
                self.user_embedding.setdefault(k, []).append(v)
    
        'news 特征'
        temp_news = [word_news_feature,  graph_news_feature]
        for _ in temp_news:
            for k, v in _.items():
                self.news_embedding.setdefault(k, []).append(v)
        
        self.news_id = list(self.news_embedding.keys())

    def recommend(self, user_id):
        merge_model = load_model(self.root + 'models/Dual_input_biinaryNN.h5')
        
        rank = dict()
        '得到目标用户画像'
        user_profile = np.hstack((self.user_embedding[user_id][0], self.user_embedding[user_id][1]))
        
        for new_id in self.news_id:
            '得到新闻向量'
            news_vec = np.hstack((self.news_embedding[new_id][0], self.news_embedding[new_id][1]))
            unknown = [user_profile.reshape((1,512)), news_vec.reshape((1,256))]
            predicted = merge_model.predict(unknown)
            #     print("\nnew_id %s CTR is: %.5f" %(new_id, predicted[0][1]), file=sys.stderr)
            '预测该用户对每个候选news的CTR'
            rank[new_id] = predicted[0][1]
        
        "推荐兴趣最高的前N个news"
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:self.N]

if __name__ == '__main__':
    start = datetime.now()
    
    user_id = '8936831'

    test = binaryDNN()
#    test.modeling()
    test.prepared()
    result = test.recommend(user_id)
    for item_id, ctr in result:
        print('item_id:%s, ctr:%.5f' % (item_id, ctr))
    print("This took ", datetime.now() - start)







