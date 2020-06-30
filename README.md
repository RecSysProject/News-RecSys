Offline：

用户-新闻行为隐反馈数据。

协同过滤

userCF：建立item-user倒排表，计算用户余弦相似度矩阵，累加用户相似度到前K个相似用户阅读集中的news上，排序前N；

itemCF：计算新闻余弦相似度矩阵，累加新闻相似度到与该用户已读新闻前k个相似的新闻集中的news上，排序前N；准确、召回率都较低；

转化为二分类，划分正负样本，统计热度、userCF和itemCF推荐度作为样本特征，针对特征连续、密集性，Xgboost分类排序；

考虑内容的重要性，用DNN模拟矩阵分解。将新闻、用户基本信息及时间上下文emdedding，新闻语料训练word2vec生成词向量，计算news特征，由用户基本信息和最近阅读news特征组成user profile。news特征和user profile双输入，分别经过一层Relu全连接后内积，接softmax分类排序