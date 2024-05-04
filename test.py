import time
import jwt
import requests

# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    #拆解api key的id和secret
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)
    
    #payload exp表示过期时间，timestamp表示当前时间 单位是毫秒
    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )
#以上都是文档示例代码 下面应该是标准的html调用
url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
headers = {
  'Content-Type': 'application/json',
  'Authorization': generate_token("0b5580be651e63ed5f130c61bd7c4d8a.76WrUqLELTaaWPwE", 1000)
}

data = {
    "model": "glm-4",
    "messages": [{"role": "user", "content": """你好，请告诉我什么是RAG"""}]
}

response = requests.post(url, headers=headers, json=data)

print("Status Code", response.status_code)
print("JSON Response ", response.json()['choices'][0]['message']['content'])
print("type of JSON Response ", type(response.json()))
