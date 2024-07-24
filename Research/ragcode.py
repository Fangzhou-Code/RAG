import os
from dotenv import load_dotenv
import random
import math
import requests
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from LocalEmbedding import LocalEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载.env文件中的环境变量
load_dotenv()

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"环境变量 '{var_name}' 未设置")
    return value

# 获取环境变量
repo_path = get_env_variable("Python_Path")

# 获取所有子文件夹
subfolders = [os.path.join(repo_path, f) for f in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, f))]
total_subfolders = len(subfolders)
print(f"总共有 {total_subfolders} 个子文件夹。")

# 计算需要随机选取的文件夹数量，向上取整
num_to_select = math.ceil(total_subfolders * 0.01)
selected_subfolders = random.sample(subfolders, num_to_select)
print(f"随机选取了 {num_to_select} 个子文件夹进行读取。")

# 加载文档
documents = []
for folder in selected_subfolders:
    print(f"正在处理文件夹: {folder}")
    loader = GenericLoader.from_filesystem(
        path=folder,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        show_progress=True
    )
    folder_documents = list(loader.lazy_load())
    print(f"文件夹 {folder} 中加载了 {len(folder_documents)} 个文档。")
    documents.extend(folder_documents)

print(f"总共加载了 {len(documents)} 个文档。")

# 并行加载文档
cpu_count = os.cpu_count()
print(f"系统中可用的 CPU 核心数: {cpu_count}")

def process_documents(docs):
    return docs

with ThreadPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(process_documents, doc) for doc in documents]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
        documents.append(future.result())

# 分割文档
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(f"总共分割了 {len(texts)} 个文本块。")

# 初始化 LocalEmbeddings 实例
model_path = get_env_variable("Local_Model_acge_text_embedding")
local_embeddings = LocalEmbeddings(
    model_path=model_path,
)

# 初始化一个空的 Chroma 数据库
final_db = Chroma(embedding_function=local_embeddings)

# 定义批量计算嵌入的函数
def compute_embeddings(batch_texts, batch_index, total_batches, progress_bar):
    logging.debug(f"开始处理批次 {batch_index + 1}/{total_batches}: {len(batch_texts)} 个文档")
    try:
        final_db.add_documents(batch_texts)
        logging.debug(f"处理完成批次 {batch_index + 1}/{total_batches}: {len(batch_texts)} 个文档")
    except Exception as e:
        logging.error(f"处理批次 {batch_index + 1}/{total_batches} 时出错: {e}")
    finally:
        progress_bar.update(1)

# 设置批处理大小
batch_size = 100
batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
total_batches = len(batches)
print(f"总批次: {total_batches}")

with ThreadPoolExecutor(max_workers=cpu_count) as executor, tqdm(total=total_batches, desc="Computing embeddings") as progress_bar:
    futures = [executor.submit(compute_embeddings, batch, i, total_batches, progress_bar) for i, batch in enumerate(batches)]
    for future in as_completed(futures):
        future.result()

print("Document database created, final_db=", final_db)

retriever = final_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8}
)

# llm = ChatOllama(
#     model="llama3",
#     temperature=0.9,
# )
chatglm_api_key = get_env_variable("ChatGLM_API_KEY_glm4")
chatglm_api_base = get_env_variable("ChatGLM_API_BASE")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=chatglm_api_key,
    openai_api_base=chatglm_api_base
)

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

questions = [
    "What is a RunnableBinding?",
    "What classes are derived from the Runnable class?",
    "What one improvement do you propose in code in relation to the class hierarchy for the Runnable class?",
]

def invoke_question(question):
    try:
        result = qa.invoke({"input": question})
        return result['answer']
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        if e.response is not None:
            print(f"响应内容: {e.response.text}")
            print(f"状态码: {e.response.status_code}")
        return None

for question in questions:
    answer = invoke_question(question)
    if answer:
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {answer} \n")
    else:
        print(f"-> **Question**: {question} \n")
        print("**Answer**: 无法获取答案，发生错误。\n")
