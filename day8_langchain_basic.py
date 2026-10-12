"""第2周 Day1：langchain 基础 PromptTemplate + Chain"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def basic_chain_demo():
    print("===Demo 1：基础 Chain ===")

    #1、初始化模型
    llm = ChatOpenAI(
        api_key = os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.7
    )

    #2、定义prompt模板
    prompt = ChatPromptTemplate.from_template(
        "你是一个{expert},请用简洁的语言回答：\n{question}"
    )

    #3、串成链：prompt->llm->解析成字符串
    chain = prompt|llm|StrOutputParser()

    #4、调用
    result = chain.invoke({
        "expert":"通信工程师",
        "question":"用一句话解释什么是OFDM"
    })

    print(result)

def multi_var_demo():
    """Demo2: 多个变量 + 不同角色"""
    print("=== Demo2 ：多个变量 + 不同角色 ===")

    llm = ChatOpenAI(
        api_key = os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.5
    )

    #2、定义prompt模板
    prompt = ChatPromptTemplate.from_template(
        "你是一个{role}.\n"
        "受众是{audiance}.\n"
        "请用{tone}的语气解释：{topic}"
    )

    chain = prompt|llm|StrOutputParser()

    result= chain.invoke({
       "role":"Python编程老师",
       "audiance":"零基础转行人员",
       "tone":"通俗易懂，带类比",
       "topic":"什么是函数(function)"
    })

    print(f"结果：{result}\n")


def raw_prompt_demo():
    print("=== Demo3：分步执行（理解内部流程) ===")

    llm = ChatOpenAI(
        api_key = os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_template("用一句话解释：{concept}")

    #分步看
    formatted = prompt.format_messages(concept = "TCP/IP三次握手")
    print(f"格式化后的消息：{formatted}")
    print(f"消息类型：{type(formatted[0])}")
    print(f"消息内容：{formatted[0].content}\n")

    #手动调用LLM
    response = llm.invoke(formatted)
    print(f"LLM原始返回：{response}")
    print(f"提取文本：{response.content}\n")




if __name__ == "__main__":
    # basic_chain_demo()
    # multi_var_demo()
    raw_prompt_demo()
