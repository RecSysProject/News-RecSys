**新闻topN推荐 + 语义搜索**

# Offline
## dataset

用户-新闻行为隐反馈数据，每条浏览记录包含四个字段：user_id、news_id、news_title、timestamp

## prepared

- 首先对没有阅读记录的用户以及没有浏览记录的新闻进行清洗

- 对用户阅读量、新闻浏览量等数据进行规范化处理

- 对新闻类别属性进行onehot编码，产生更多特征

## modeling

 - 协同过滤，userCF：建立item-user倒排表，计算用户余弦相似度矩阵，累加用户相似度到前K个相似用户阅读集中的news上，排序前N；itemCF：计算新闻余弦相似度矩阵，累加新闻相似度到与该用户已读新闻前k个相似的新闻集中的news上，排序前N；准确、召回率都较低
 - 转化为二分类，划分正负样本，统计热度、userCF和itemCF推荐度作为样本特征，针对特征连续、密集性，Xgboost分类排序       
 - 旨在降低训练复杂度，同时加强特征交叉能力，将连续特征离散化后，采用FM模型，学习每个特征的k维隐权重向量 
 - 考虑内容的重要性，将新闻、用户基本信息及时间上下文emdedding，新闻语料负采样训练word2vec生成词向量，计算news特征，由用户基本信息和最近阅读的news特征组成user profile。用DNN模拟矩阵分解，news特征和user profile双输入，分别经过一层Relu全连接后内积，接softmax分类排序
 - query拓展后，ANN（近似最近邻）检索语义相近的新闻

## evaluation

- 协同过滤给出的是Precision、Recall、Coverage、Popularity四项指标衡量模型的质量
- 二分类模型采用的是AUC

# Online
部署线上推荐服务，restfulAPI接口

 - 冷启动采用热门新闻及基于用户基本信息的兴趣分类
 - 先对userCF、itemCF、热度三路召回，过滤劣质新闻后，DNN排序

# attention

UserCF算法中，由于用户数量多，生成的相似性矩阵也大，会占用比较多的内存。

ItemCF算法中，每次推荐都需要找出一个用户的所有新闻，再为每一条新闻找出最相似的新闻，运算量比UserCF大，因此推荐的过程比较慢。
