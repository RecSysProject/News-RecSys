#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json

# api路径
#url="http://0.0.0.0:5555/queryExpan"
url="http://106.12.83.14:5555/queryExpan"

# 传入参数
parms = {
    'query': '板楼'  # 发送给服务器的内容
}

headers = {
    'User-agent': 'none/ofyourbusiness',
    'password': 'Eggs'
}

res = requests.post(url, data=parms,headers=headers)  # 发送请求

text = res.text
print(json.loads(text))
