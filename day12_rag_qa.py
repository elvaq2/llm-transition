import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#========== 1、加载 Day4 的向量库===========
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",cache_folder="./models")#初始化嵌入模型，文本转向量靠这个
vectorstore = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
retriver = vectorstore.as_retriever(search_kwargs={"k":2})#将向量库包装成检索组件，可以放进LCEL管道
#返回列表对象

#========== 2、初始化大模型===========
llm = ChatOpenAI(
    # model = "deepseek-chat",
    # openai_api_key = os.getenv("OPENAI_API_KEY"),
    # openai_base_url = os.getenv("OPENAI_BASE_URL")
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="glm-4-flash",
)
res = llm.invoke("你好")
print(res.content)
#========== 3、构造 RAG Prompt(核心：注入context)===========
template = """你是一个专业问答助手，请**仅基于**以下检索到的上下文回答问题。
如果上下文没有答案，请直接说“根据已知信息无法回答问题”，不要编造

上下文：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)

#========== 4、格式化检索结果（把Document列表转换成字符串)===========
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#==========5、LCEL现代RAG链（检索-注入-生成)===========
rag_chain = (
    {"context":retriver|format_docs,"question":RunnablePassthrough()}#retriever返回的是列表对象，prompt接收字符串，所以必须将retriever输出结果转换为字符串，注入prompt的占位符“context”
    |prompt
    |llm
    |StrOutputParser()
)
# LangChain 规则：在管道 `|` 里面，直接放字典，会自动包装成 `RunnableParallel`
# 其实是：
# RunnableParallel({
#     "context":retriver|format_docs,
#     "question":RunnablePassthrough()
# })
# chain.invoke(param)中的param同时传递给RunnableParallel字典里的每一个key分支，每一个key对应的Runnable独立计算


#==========6、测试（带DAY3记忆的进阶版在下方===========
if __name__ == "__main__":
    questions = [
        "elva之前是做什么工作的?",
        "RAG是什么意思？",
        "她为什么学langchain？",
        "今天天气怎么样？"
    ]

    for q in questions:
        print(f"\n问：{q}")
        ans = rag_chain.invoke(q)
        print(f"答：{ans}")



