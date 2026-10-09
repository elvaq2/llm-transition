import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def chat_raw(prompt:str, model:str = "glm-4-flash",temperature:float = 0.7):
    """用requests 裸调智谱 API"""

    #1、拼header
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    #2、拼body
    payload = {
        "model":model,
        "messages":[{"role":"user","content":prompt}],
        "temperature":temperature,
        "max_tokens":512
    }

    #3、发POST请求
    print(f"POST {BASE_URL}")
    print(f"模型：{model}")
    print("发送中...")

    try:
        response = requests.post(
            BASE_URL,
            headers = headers,
            json = payload,
            timeout=30
        )

        #4、检查HTTP状态码
        response.raise_for_status()# 如果状态码不是200，抛出HTTPError异常

        #5、检查HTTP状态码
        result = response.json()# 把接口返回的json字符串，转成python字典
        return result["choices"][0]["message"]["content"]
        #result["choices"]`：拿到**choices 列表**，大模型返回的候选回答数组
        #[0]：取列表第**1 个**回答（下标从 0 开始，默认只返回 1 条）
        #["message"]：取出这条回答里消息对象
        #["content"]：取出模型生成的文本内容

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误:{e}")
        print(f"返回内容：{response.text}")
        
        return None

    except requests.exceptions.Timeout:
        print("请求超时")
        return None

    except requests.exceptions.ConnectionError:
            print("网络连接失败")
            return None

    except KeyError as e:
        print(f"返回格式异常，找不到字段:{e}")
        print(f"完整返回：{result}")
        
        return None

if __name__ =="__main__":
     #测试
    reply = chat_raw("用一句话解释什么是HTTP POST请求")
    if reply:
        print(f"\nAI 回复：{reply}")
    else:
        print("\n调用失败")
