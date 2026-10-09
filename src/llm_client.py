import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.config import ModelConfig

load_dotenv()

class LLMClient:
    def __init__(self, config_path="config.json"):
        #读配置
        # with open("config.json","r",encoding="utf-8")as f:
        #     self.config = json.load(f)#load成一个词典

        self.config = ModelConfig.from_json(config_path)#返回一个ModelConfig对象

        #初始化客户端
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("未找到ZHIPU_API_KEY，请检查.env文件")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )

    #1、用户输入；2、config配置文件读入；3、默认值；
    def chat(self,prompt:str,max_tries=3,**kwargs):
        """带重试的对话方法"""
        #合并配置：kwargs优先级高于config.json
        # model = kwargs.get("model",self.config.get("model","glm-4-flash"))
        model = kwargs.get("model",self.config.model)
        # temperature = kwargs.get("temperature",self.config.get("temperature",0.7))
        temperature = kwargs.get("temperature",self.config.temperature)
        # max_tokens = kwargs.get("max_tokens",self.config.get("max_tokens",512))
        max_tokens = kwargs.get("max_tokens",self.config.max_tokens)

        for attempt in range(1,max_tries+1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role":"user","content":prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"第{attempt}次调用失败：{e}")
                if attempt < max_tries:
                    wait = 2**attempt #指数退避：2s,4s,8s
                    print(f"等待{wait}s 后重试...")
                    time.sleep(wait)

                else:
                    return f"调用失败，已重试{max_tries}次，{e}"

