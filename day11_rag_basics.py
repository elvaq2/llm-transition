import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

#===============1、加载文档 =======================
def load_doc():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir,"data","elva_notes.txt")
    loader = TextLoader(txt_path,encoding="utf-8")
    docs = loader.load()
    print(f"加载了{len(docs)}个文档")
    print(f"    内容预览：{docs[0].page_content[:50]}...")
    return docs

#===============2、切片 =======================
def split_doc(docs):
    spilitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20,
        separators = ["\n\n","\n","。"," ",""]
    )

    chunks = spilitter.split_documents(docs)
    print(f"切成了{len(chunks)}块")
    for i,c in enumerate(chunks):
        print(f"    [{i}]{c.page_content[:40]}...")

    return chunks

#===============3、向量化 =======================
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model = "BAAI/bge-small-zh-v1.5",#小型模型
        cache_folder = "./models",#模型下载到这个路径
        model_kwargs = {"trust_remote_code":True},
        encode_kwargs = {"normalize_embeddings":True}
    )

    vectorstores = FAISS.from_documents(chunks,embeddings)
    print("向量库构建完成")
    return vectorstores

#===============4、相似度检索 =======================
def search(vectorstores,query):
    print(f"\n提问：{query}")
    results = vectorstores.similarity_search(query,k=2)
    for i,r in enumerate(results):
        print(f"    [{i+1}]{r.page_content.strip()}")
    return results


#==================入口=================
if __name__ == "__main__":
    docs = load_doc()
    chunks = split_doc(docs)
    vs = build_vectorstore(chunks)

    search(vs,"elva之前是做什么工作的？")
    search(vs,"RAG是什么意思")
    search(vs,"有哪些向量数据库")
