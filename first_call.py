import os
from openai import OpenAI
from dotenv import load_dotenv

#1、加载.env
load_dotenv()
key = os.getenv("ZHIPU_API_KEY")
print("key shi:",repr(key))
#2、创建客户端（智谱）
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    # key = os.getenv("ZHIPU_API_KEY")
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

#3、发请求
resp=client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role":"system","content":"你是一个建筑师转的大模型专家"},
        {"role":"user","content":"请用一句话形容大模型"}
    ]   
)

#4、打印结果
print("模型回复：")
print(resp.choices[0].message.content)