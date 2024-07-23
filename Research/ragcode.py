from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from LocalEmbedding import LocalEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv() # 加载.env文件中的环境变量

def get_env_variable(var_name: str) -> str:
    """
    获取环境变量的值，如果环境变量不存在则抛出异常

    Args:
    var_name (str): 环境变量名称

    Returns:
    str: 环境变量的值

    Raises:
    ValueError: 如果环境变量未设置
    """
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"环境变量 '{var_name}' 未设置")
    return value


# Clone
repo_path = get_env_variable("Repo_Path")
# Load
loader = GenericLoader.from_filesystem(
    repo_path + "/libs/core/langchain_core",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
print(len(documents))

# Splitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(len(texts))

# RetrievalQA
model_path = get_env_variable("Local_Model_acge_text_embedding")
# 初始化 LocalEmbeddings 实例，使用本地模型的路径
local_embeddings = LocalEmbeddings(
    model_path=model_path,  # 本地模型文件的路径
)
# 初始化 LocalEmbeddings 实例，使用本地模型和分词器的路径
# model_path = get_env_variable("Local_Model_acge_text_embedding")
# tokenizer_path = get_env_variable("Local_Tokenizer_Path")
# local_embeddings = LocalEmbeddings(
#     model_path=model_path,  # 本地模型文件的路径
#     tokenizer_path=tokenizer_path  # 本地分词器的路径
# )



# 使用 LocalEmbeddings 和 Chroma 创建文档数据库
db = Chroma.from_documents(texts, local_embeddings)
print("Document database created, db=.", db)



# 创建检索器，可以使用 'mmr' 或 'similarity' 检索类型
retriever = db.as_retriever(
    search_type="mmr",  # 或 "similarity"
    search_kwargs={"k": 8}  # k 表示检索的文档数量
)



# 调用chatglm
# chatglm_api_key = get_env_variable("ChatGLM_API_KEY_glm4")
# chatglm_api_base = get_env_variable("ChatGLM_API_BASE")
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(
#     temperature=0.95,
#     model="glm-4",
#     openai_api_key=chatglm_api_key,
#     openai_api_base=chatglm_api_base
# )

# 调用本地ollama
# llm = ChatOllama(
#     model="llama3",
#     temperature=0.9,
#     # other params...
# )

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)

question = "What is a RunnableBinding?"
result = qa.invoke({"input": question})
print("answer=", result["answer"])