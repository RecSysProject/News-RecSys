#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from sqlWorkflow import Workflow
sys.path.append("..")       #导入平级目录模块
from OffLine.userCF import userCF
from flask import Flask, g
from flask_restful import reqparse, Api, Resource

# Flask相关变量声明
app = Flask(__name__)
api = Api(app)

# RESTfulAPI的参数解析 -- put / post参数解析
parser_put = reqparse.RequestParser()
parser_put.add_argument("user_id", type=str, required=True, help="need user data")
#parser_put.add_argument("pwd", type=str, required=True, help="need pwd data")

# 功能方法
def Rec_News(argv_):
    result = []
    workflow = Workflow()
    rec = userCF()
#    test.split_dataset()
#    test.user_sim()
    rec.prepared()
    temp = rec.recommend(argv_)
    
    for item_id, ctr in temp:
        middle = workflow.sqlSearch(item_id)
        middle.setdefault('ctr', round(ctr,4))
        result.append(middle)
#        result.setdefault(item_id, middle)
    return result

# 操作（post / get）资源列表
class TodoList(Resource):
    
    def post(self):
        args = parser_put.parse_args()
        
        # 构建新参数
        user_id = args['user_id']
        # pwd = args['pwd']
        print(user_id)
        # 调用方法Rec_News
        info = Rec_News(user_id)
    
        # 资源添加成功，返回201
        return info, 201

# 设置路由，即路由地址为http://106.12.83.14:7777/recommend
api.add_resource(TodoList, "/recommend")

if __name__ == "__main__":
    app.run(host='0.0.0.0',#任何ip都可以访问
            port=7777,#端口
            debug=True)
