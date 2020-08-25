import re
import json
import jieba
from jieba import analyse
import numpy as np
import scipy.io as sio
from snownlp import SnowNLP
from datetime import datetime

import TencentYoutuyun
from TencentYoutuyun import *
from gensim.models import word2vec

class userProfile(object):
    
    def __init__(self):
        appid = '10109383'
        userId = '875974254'
        secretId = 'AKIDd3D8rKrzCAsKXXKn8E5i6EAsLYVCuoiP'
        secretKey = 'ZtwjGYbP1PYT9anmV3MRGrCKDuPffOr4'
        endPoint = TencentYoutuyun.conf.API_YOUTU_END_POINT
        self.youtu = TencentYoutuyun.YouTu(appid, secretId, secretKey, userId, endPoint)
        
        self.root = '/Users/tung/Python/PersonalProject/NewsRecommend/'
        self.imgFile = self.root + 'On-line/user-profile/face.jpg'
        self.Nickname = '木冉'
        self.signature = "心向阳光 野蛮成长☀️"
        self.A = {'sex': 'Unknow'}
        self.B = {'sex': 'Male'}
        self.C = {'sex': 'Female'}
        self.age = 26
        self.row = {'NickName':'alice', 'Province':'beiijng', 'City':'beijign'}
    
    '头像信息'
    #人脸检测
    def detectFace(self,image):
        try:
            retocr = self.youtu.DetectFace(image)
            return len(retocr['face'])>0
        except Exception as e:
            return false
    #图片关键词
    def extractTags(self,image):
        try:
            retocr = self.youtu.imagetag(image)
            return retocr['tags']
        except Exception as e:
            return None
    #头像标签
    def img_tags(self, imgFile):
        image_tags =''
        result = self.extractTags(imgFile)
        if(result != None and len(result)>=2):
            image_tags = result[0]['tag_name'].encode('iso8859-1').decode('utf-8')+ ' ' + result[1]['tag_name'].encode('iso8859-1').decode('utf-8')
        return image_tags
    #是否用人脸
    def use_face(self, imgFile):
        IFface = self.detectFace(imgFile)
        if IFface:
            return ''.join('外向')
        return ''.join('内向')

    '个性签名'
    #情绪倾向
    def get_emotion(self, Nickname, signature):
        if(signature != None):
            signature = signature.strip().replace('span', '').replace('class', '').replace('emoji', '')
            signature = re.sub(r'1f(\d.+)','',signature)
            if(len(signature)>0):
                nlp = SnowNLP(signature)
                emotions = nlp.sentiments
                if emotions > 0.66:
                    return '积极'
                elif  emotions>=0.33 and emotions<=0.66:
                    return '理性'
                return '消极'
    #关键词
    def sign_tags(self, signature):
        if(signature != None):
            signature = signature.strip().replace('span', '').replace('class', '').replace('emoji', '')
            signature = re.sub(r'1f(\d.+)','',signature)
            if(len(signature)>0):
                tags = ' '.join(jieba.analyse.extract_tags(signature,5))
            return tags

    '性别'
    def gender_aware(self, sex):
        if sex == 'Male':
            return '硬' +' '+'经济' +' '+ '政治' +' '+ '科技'
        elif  sex == 'Female':
            return '软' +' '+  '娱乐' +' '+ '服务' +' '+ '社会'
        return

    '年龄'
    def age_group(self, age):
        if age > 38:
            return '中年'
        elif  age < 20:
            return '少年'
        return  '青年'

    '位置'
    def region_tende(self, row):
        Province = row['Province']
        return Province

    '词嵌入表示'
    def transform_to_matrix(self, x, vec_size=128):
        matrix = []
        #加载已训练好的word2vec模型
        word_model = word2vec.Word2Vec.load(self.root + 'word-embedding/sougouCS_wordVec')
        for sen in x:
            try:
                matrix.append(word_model[sen].tolist())
            except:
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0] * vec_size)
        return matrix

    '生成静态画像'
    def staticProfile(self):
        img_tags = self.img_tags(self.imgFile)
        img_character = self.use_face(self.imgFile)
        sign_emotion = self.get_emotion(self.Nickname, self.signature)
        sign_tags = self.sign_tags(self.signature)
        gender_aware = self.gender_aware(self.B['sex'])
        age_group = self.age_group(self.age)
        region_tende = self.region_tende(self.row)
        '静态画像'
        user_profile = img_tags +' '+img_character +' '+ sign_emotion +' '+ sign_tags +' '+ gender_aware +' '+ age_group
        user_profile = user_profile.split(' ')
        user_profile_vec = self.transform_to_matrix(user_profile)
        #保存到本地
        #sio.mmwrite(self.root + 'user_profileMatrix', user_profile_vec)
        
        return user_profile_vec
    #本地读画像
    #user_profileMatrix = sio.mmread(self.root + 'user_profileMatrix')
    #user_profileMatrix.shape

if __name__ == '__main__':
    start = datetime.now()
#    root = '/Users/tung/Python/PersonalProject/NewsRecommend/'
#    imgFile = root + 'On-line/user-profile/face.jpg'
#    Nickname = '木冉'
#    signature = "心向阳光 野蛮成长☀️"
#    B = {'sex': 'Male'}
#    age = 26
#    row = {'NickName':'alice', 'Province':'beiijng', 'City':'beijign'}

    test = userProfile()
#    img_tags = test.img_tags(imgFile)
#    print(img_tags)
#    img_character = test.use_face(imgFile)
#    print(img_character)
#    sign_emotion = test.get_emotion(Nickname, signature)
#    print(sign_emotion)
#    sign_tags = test.sign_tags(signature)
#    print(sign_tags)
#    gender_aware = test.gender_aware(B['sex'])
#    print(gender_aware)
#    age_group = test.age_group(age)
#    print(age_group)
#    region_tende = test.region_tende(row)
#    print(region_tende)
#
#    '静态画像'
#    user_profile = img_tags +' '+img_character +' '+ sign_emotion +' '+ sign_tags +' '+ gender_aware +' '+ age_group
#    user_profile = user_profile.split(' ')
#    print(user_profile)

    user_profile_vec = test.staticProfile()
    print('user_profile vec shape is', np.shape(user_profile_vec))
    
    print("This took ", datetime.now() - start)
