#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import json
import pymysql
import _pickle as cPickle
from urllib import parse
from urllib import request
from urllib.request import urlopen

root = '/Users/tung/Python/PersonalProject/NewsRecommend/On-line'

'每天更新新闻到数据库'
#获取新闻，添加入数据库13类*40个 = 520条新闻
channel = ['头条','财经','体育','娱乐','军事','教育','科技','NBA','股票','星座','女性','健康','育儿']
data = {}
data["appkey"] = "1358c9524b37b9ba"
data["channel"] = "头条"  #新闻频道(头条,财经,体育,娱乐,军事,教育,科技,NBA,股票,星座,女性,健康,育儿)
data["start"] = "0"
data["num"] = "40"
url_values = parse.urlencode(data)
url = "https://api.jisuapi.com/news/get" + "?" + url_values
print(url)

request = request.Request(url)
result = urlopen(request)
jsonarr = json.loads(result.read())

if jsonarr["status"] != u"0":
    print( jsonarr["msg"])
#     exit()
result = jsonarr["result"]
print(result["channel"],result["num"])

#连接数据库
#1打开数据库连接
conn = pymysql.connect(host = '106.12.83.14', user = 'ping', passwd = 'mima123456', port=3306,
                       charset='utf8', autocommit=True) #utf-8编码，否则中文有可能会出现乱码。

#2创建一个游标,用来给数据库发送sql语句
cursor=conn.cursor()
#3对数据库进行增删改查操作
#选择需要的数据库
conn.select_db('test1')

#数据库内增加新数据
for val in result["list"]:
    sql= "insert into news(title,time,src,cat,pic,content,url,weburl) values (%s, %s, %s, %s, %s, %s, %s, %s)" #插入数据库
    #     content = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".encode('utf-8').decode('utf-8'), "".encode('utf-8').decode('utf-8'), val["content"])
    #     content = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", content)
    cursor.execute(sql,(val["title"],val["time"],val["src"],val["category"],val["pic"], val["content"], val["url"],val["weburl"]))
    print("标题:{0}时间:{1}".format(val["title"],val["time"]))


'获取推荐候选集'
#最新的520*3天=1560条新闻
#SQL语句
# sql="select * from news where 1=1 limit 650"  #前13*50条
sql="select * from news order by id desc limit 0,1560" #后13*80条
try:
    cursor.execute(sql)
    candidate = cursor.fetchall() #获取全部结果集。 fetchone 查询第一条数据，返回tuple类型。
    if not candidate: #判断是否为空。
        print("数据为空！")
    else:
        for row in candidate:
            ID = row[0]
            title = row[1]
            time = row[2]
            src = row[3]
            cat = row[4]
            pic = row[5]
            content = row[6]
            url = row[7]
            weburl = row[8]
# 打印结果
#         print("id:{0}标题:{1}时间:{2}来源:{3}标签:{4}图片:{5}内容:{6}url:{7}weburl:{8}".format(ID,title,time,src,cat,pic,content,url,weburl))

except Exception as e:
    conn.rollback()  #如果出错就会滚数据库并且输出错误信息。
    print("Error:{0}".format(e))
finally:
    conn.close()#关闭数据库。

print( '获取候选新闻的数据量为：', len(candidate))
cPickle.dump( candidate, open(root + 'candidate.pkl', 'wb')) #tuple对象持久化


'从本地读入，准备召回'
#[1]标题、[2]时间、[4]分类、[6]内容
candidate = cPickle.load( open(root + 'candidate.pkl','rb') ) #读入数据
print('候选新闻样本：', candidate[20])