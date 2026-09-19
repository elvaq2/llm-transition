# import os

# class LLMClient:
#     def __init__(self,api_key:str,base_url:str="https://open.bigmodel.cn/api/paas/v4/"):
#         self.api_key=api_key
#         self.base_url = base_url
#         print(f"客户端初始化：{self.base_url}")

#     def chat(self,prompt:str,*args,**kwargs):
#         #args可接收system prompt等
#         #kwargs透传temperature/top-p/model等
#         config = {
#             "model":kwargs.get("model","glm-4-flash"),
#             "temperature":kwargs.get("temperature",0.7),
#             "max_tokens":kwargs.get("max_tokens",512),
#         }
#         print(f"发送请求：{prompt}")
#         print(f"配置：{config}")
#         print(f"额外参数：{kwargs}")

#         return f"AI 回复：{prompt}"


# client = LLMClient(api_key=os.getenv("ZHIPU_API_KEY"))
# reply = client.chat(
#     "什么是OFDM",
#     "你是通信专家",
#     model = "glm-4-plus",temperature=0.3
# )
# print(reply)


def uart_send(*args,**kwargs):
    print(f"args: {args}")
    print(f"kwargs: {kwargs}")

# uart_send("as","ef",kye1="value1",key2="value2")

class SensorClient:
    def __init__(self,sensor_id):
        self.sensor_id = sensor_id

    def read(self,*args,**kwargs):
        print(f"读取传感器号0x{self.sensor_id:02}")
        print(f"命令args{args}")
        print(f"配置kwargs{kwargs}")

        return {"sensor_id":self.sensor_id,"config":kwargs}
        sample_rate = kwargs.get(sample_rate,1000)
        gain = kwargs.get(gain,0.5)

sensor = SensorClient(sensor_id=0x0a)
data = sensor.read(
    "CH1","CH2",
    sample_rate = 1000,
    gain = 0.5,
    resolution = 12
)
print("返回数据：",data)

    