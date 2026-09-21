# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("ZHIPU_API_KEY")

# print("API Key 已加载：",api_key[:8]+"...."if api_key else "未找到")

import json

with open("config.json","r",
          encoding="utf-8")as f:config = json.load(f)
print("当前配置：",config)
print("模型：",config[])