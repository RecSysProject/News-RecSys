import requests
import json

# api路径
url="http://0.0.0.0:7777/search"
#url="http://106.12.83.14:7777/search"

# 传入参数
parms = {
    'query': '经济'  # 发送给服务器的内容
}

headers = {
    'User-agent': 'none/ofyourbusiness',
    'password': 'Eggs'
}

res = requests.post(url, data=parms,headers=headers)  # 发送请求

text = res.text
print(json.loads(text))

