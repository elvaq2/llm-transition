from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory as CommunityChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os


load_dotenv()


def demo1_no_memory():
    """Demo 1：无记忆（对比用）"""
    print("=== Demo 1：无记忆（AI 记不住） ===")

    llm = ChatOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template("{input}")

    # 第一轮
    r1 = (prompt | llm).invoke({"input": "我叫Elva，今年28岁"})
    print(f"你: 我叫Elva，今年28岁")
    print(f"AI: {r1.content}\n")

    # 第二轮（没有传历史，AI 不知道上一轮说了啥）
    r2 = (prompt | llm).invoke({"input": "我叫什么名字？多大了？"})
    print(f"你: 我叫什么名字？多大了？")
    print(f"AI: {r2.content}\n")


def demo2_buffer_memory():
    """Demo 2：ConversationBufferMemory（完整记忆）"""
    print("=== Demo 2：BufferMemory（AI 记住了） ===")

    llm = ChatOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.7
    )

    # 创建记忆
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

    # 带 MessagesPlaceholder 的 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，请根据对话历史回答问题。"),
        MessagesPlaceholder(variable_name="history"),  # 历史消息插入点 第一轮没有就空着
        ("human", "{input}")
    ])

    chain = prompt | llm

    # 第一轮
    user_input_1 = "我叫Elva，在通信行业做了5年，现在想转行做大模型"
    print(f"你: {user_input_1}")

    response_1 = chain.invoke({
        "input": user_input_1,
        "history": memory.chat_memory.messages
    })
    print(f"AI: {response_1.content}\n")

    # 把这一轮存入记忆
    memory.chat_memory.add_user_message(user_input_1)
    # memory.chat_memory.add_ai_message(response_1.content)

    # 第二轮（AI 能看到历史了）
    user_input_2 = "你还记得我叫什么名字吗？我之前是做什么的？"
    print(f"你: {user_input_2}")

    response_2 = chain.invoke({
        "input": user_input_2,
        "history": memory.chat_memory.messages#方才添加的user/ai message
    })
    print(f"AI: {response_2.content}\n")

    # 第三轮（继续记住）
    user_input_3 = "你觉得我转行最大的优势是什么？"
    print(f"你: {user_input_3}")

    response_3 = chain.invoke({
        "input": user_input_3,
        "history": memory.chat_memory.messages
    })
    print(f"AI: {response_3.content}\n")

    # 看看记忆里存了什么
    print("--- 记忆内容 ---")
    for msg in memory.chat_memory.messages:
        print(f"  {msg.type}: {msg.content[:500]}...")


def demo3_memory_with_chain():
    """Demo 3：用 RunnableWithMessageHistory（更优雅的写法）"""
    print("\n=== Demo 3：RunnableWithMessageHistory（自动管记忆） ===")

    llm = ChatOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个Python编程老师，用简洁的方式回答问题。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    # 用 RunnableWithMessageHistory 自动管理
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_classic.memory import ChatMessageHistory

    # 会话存储（简单字典，实际项目可用 Redis/数据库）
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,#指定加载那一段历史，塞进prompt里的MessagePlaceHolder
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": "user_001"}}

    # 第一轮
    r1 = chain_with_history.invoke(
        {"input": "什么是 Python 的装饰器？"},
        config=config
    )
    print(f"你: 什么是 Python 的装饰器？")
    print(f"AI: {r1.content[:100]}...\n")

    # 第二轮（自动带上历史）
    r2 = chain_with_history.invoke(
        {"input": "能给我写个最简单的例子吗？"},
        config=config
    )
    print(f"你: 能给我写个最简单的例子吗？")
    print(f"AI: {r2.content[:100]}...\n")

    print(f"当前会话历史条数: {len(store['user_001'].messages)}")


if __name__ == "__main__":
    # demo1_no_memory()
    # demo2_buffer_memory()
    demo3_memory_with_chain()
    print("\n=== Day 3 完成 ===")