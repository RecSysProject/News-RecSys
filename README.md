News-Recommend-System
=========================================================================================

### 针对新闻分发内容质量差、过度相似现象，为用户实时推荐个性化优质新闻。负责TopN推荐算法的设计改进，召回、过滤、排序


### 构建特征矩阵news、user profile相似度、userCF推荐度、热度；新闻语料训练word2vec生成word向量，拼接news特征；由用户基本信息和已读news特征组成user profile矩阵，计算与候选news相似度；通过user profile相似用户集合，计算userCF推荐度；提取近期热词表，热度量化指标tf-idf引入周期性衰减，热词与阅读量表征新闻热度；朴素贝叶斯分类过滤劣质内容


### 生成推荐，首先用faiss索引，对user profile相似度、userCF推荐度、热度三路召回，针对特征连续、密集性，Xgboost排序推荐，覆盖率低；后将用户及上下文emdedding，三层Relu输出与news矩阵内积，接softmax召回