# # Day 2：字符串 / f-string / 切片

# # ===== 1. 字符串基础 =====
# system_prompt: str = "You are a helpful assistant."
# user_prompt: str = "Explain LSTM in one sentence."
# print(system_prompt)
# print(user_prompt)

# # ===== 2. f-string（拼 prompt 最常用）=====
# model: str = "glm-4-flash"
# temperature: float = 0.7
# max_tokens: int = 512

# prompt = f"""
# Model: {model}
# Temperature: {temperature}
# Max Tokens: {max_tokens}
# """
# print(prompt)

# # ===== 3. 切片（处理模型输出极有用）=====
# text: str = "Hello, Large Language Model!"
# print("全文：", text)
# print("从索引 7 开始：", text[7:])
# print("前 5 个字符：", text[:5])
# print("最后 6 个字符：", text[-6:])

# # ===== 4. 字符串常用方法（清洗模型输出）=====
# raw_output: str = "  Sure, here is the answer. \n\n"
# print("原始：", raw_output)
# clean: str = raw_output.strip()          # 去首尾空白
# clean = clean.replace("Sure, ", "")      # 替换无用前缀
# print("清洗后：", clean)

# # ===== 5. 实战：拼一条真实 prompt =====
# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": user_prompt}
# 
# print("\n最终 messages：")
# for msg in messages:
#     print(f"{msg['role']}: {msg['content']}")

string:str=" blankstrip  "
string = string.strip()
print(string)
string = string.replace("strip"," ")
print(string)
