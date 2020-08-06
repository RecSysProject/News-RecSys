# -*- coding: utf-8 -*-
import sys
sys.path.append("..")       #导入平级目录模块
from OffLine.semanticSearch import search
from flask import Flask, g
from flask_restful import reqparse, Api, Resource

# Flask相关变量声明
app = Flask(__name__)
api = Api(app)

# RESTfulAPI的参数解析 -- put / post参数解析
parser_put = reqparse.RequestParser()
parser_put.add_argument("query", type=str, required=True, help="need user data")
#parser_put.add_argument("pwd", type=str, required=True, help="need pwd data")

# 功能方法
def Search_News(argv_):
    result = {}
    test = search()
    temp = test.FlatL2(argv_)
    for news_id, title in temp:
        result.setdefault(news_id, title)
    return result

# 操作（post / get）资源列表
class TodoList(Resource):
    
    def post(self):
        args = parser_put.parse_args()
        
        # 构建新参数
        query = args['query']
#        pwd = args['pwd']
        print('input query:%s' % query)
        # 调用方法semantic_search
        info = Search_News(query)
    
        # 资源添加成功，返回201
        return info, 201

# 设置路由，即路由地址为http://106.12.83.14:7777/recommend
api.add_resource(TodoList, "/search")

if __name__ == "__main__":
    app.run(host='0.0.0.0',#任何ip都可以访问
            port=7777,#端口
            debug=True)