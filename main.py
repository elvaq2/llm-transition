from src.llm_client import LLMClient

def main():
    print("===LLM Client Demo")

    try:
        client = LLMClient(config_path = "config.json")
        print("客户端初始化成功")
    except Exception as e:
        print(f"初始化失败：{e}")
        return

    #真实对话
    prompt = "用一句话解释什么是TCP/IP三次握手"
    print(f"\n问题：{prompt}")
    print("思考中...")

    reply = client.chat(prompt,temperature = 0.3)
    print(f"\nAI回复：{reply}")

    #测试重试（故意传错模型名）
    print("\n--- 测试错误处理 ---")
    bad_reply = client.chat("hi",model = "nonexistent-model")
    print(bad_reply)

if __name__=="__main__":
    main()
                            
        
    
