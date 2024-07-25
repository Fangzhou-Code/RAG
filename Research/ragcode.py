from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import SentenceTransformer, AutoModelForSequenceClassification, AutoTokenizer
from LocalEmbedding import LocalEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import random
import math
import requests
from langchain_community.retrievers import BM25Retriever
import json
import torch


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

# 加载.env文件中的环境变量
load_dotenv() 

# 加载文档
repo_path = get_env_variable("Python_Path")

# 获取所有子文件夹
subfolders = [os.path.join(repo_path, f) for f in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, f))]
total_subfolders = len(subfolders)
print(f"总共有 {total_subfolders} 个子文件夹。")

# 创建 GenericLoader 实例，并显示进度条
documents = []
selection_ratio = 0.01  # 可以根据需要调整
for folder in subfolders:
    print(f"正在处理文件夹: {folder}")
    loader = GenericLoader.from_filesystem(
        path=folder,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),  # parser_threshold: Minimum lines needed to activate parsing (0 by default).
        show_progress=True  # 显示进度条
    )
    all_documents = list(loader.lazy_load())
    num_to_select = math.ceil(len(all_documents) * selection_ratio)
    selected_documents = random.sample(all_documents, num_to_select)
    documents.extend(selected_documents)
    print(f"文件夹 {folder} 中加载了 {len(selected_documents)} 个文档（总计 {len(all_documents)} 个文档中的 {selection_ratio*100}%）。")

print(f"总共加载了 {len(documents)} 个文档。")

# 并行加载文档
document_list = list(documents)
print(f"len(document_list): {len(document_list)}")

# 打印 CPU 核心数
cpu_count = os.cpu_count()
print(f"系统中可用的 CPU 核心数: {cpu_count}")
with ThreadPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(lambda doc: doc, doc) for doc in document_list]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading documents"):
        documents.append(future.result())

# Splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(len(texts))





# llm
chatglm_api_key = get_env_variable("ChatGLM_API_KEY_glm4")
chatglm_api_base = get_env_variable("ChatGLM_API_BASE")
llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=chatglm_api_key,
    openai_api_base=chatglm_api_base
)
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



# 示例调用方法
def invoke_question(qa, question):
    try:
        result = qa.invoke({"input": question})  # 这里假设 qa 是一个已经定义的对象
        return result
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        if e.response is not None:
            print(f"响应内容: {e.response.text}")
            print(f"状态码: {e.response.status_code}")
        return None
    
# 定义批量计算嵌入的函数
def compute_embeddings(db, batch_texts, batch_index, total_batches, progress_bar):
    logging.debug(f"开始处理批次 {batch_index + 1}/{total_batches}: {len(batch_texts)} 个文档")
    try:
        # 使用 add_documents 方法将文档添加到 Chroma 数据库中
        db.add_documents(batch_texts)
        logging.debug(f"处理完成批次 {batch_index + 1}/{total_batches}: {len(batch_texts)} 个文档")
    except Exception as e:
        logging.error(f"处理批次 {batch_index + 1}/{total_batches} 时出错: {e}")
    finally:
        progress_bar.update(1)

questions = [
    "What is a RunnableBinding?",
]



# 创建 acge 检索器
acge_model_path = get_env_variable("Local_Model_acge_text_embedding")
# 初始化 LocalEmbeddings 实例，使用本地模型的路径
local_embeddings_acge = LocalEmbeddings(
    model_path=acge_model_path,  # 本地模型文件的路径
)
# 初始化一个空的 Chroma 数据库
acge_db = Chroma(embedding_function=local_embeddings_acge)
# 设置批处理大小
batch_size = 100  # 批处理大小
batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
total_batches = len(batches)
print(f"总批次 {total_batches}")
# 并行计算嵌入并合并结果
with ThreadPoolExecutor(max_workers=cpu_count) as executor, tqdm(total=total_batches, desc="Computing embeddings") as progress_bar:
    futures = [executor.submit(compute_embeddings, acge_db, batch, i, total_batches, progress_bar) for i, batch in enumerate(batches)]
    for future in as_completed(futures):
        future.result()
print("Document database created, acge_db=", acge_db)
# 获取索引中的文档数量
num_documents_in_index = len(acge_db._collection.get()["documents"])
print(f"索引中的文档数量:{num_documents_in_index}")
# 创建检索器，可以使用 'mmr' 或 'similarity' 检索类型
acge_retriever = acge_db.as_retriever(
    search_type="mmr",  # 或 "similarity"
    search_kwargs={"k": 8, 'fetch_k': min(20, num_documents_in_index)},  # k: Amount of documents to return (Default: 4); fetch_k: Amount of documents to pass to MMR algorithm
)
acge_retriever_chain = create_history_aware_retriever(llm, acge_retriever, prompt)
acge_qa = create_retrieval_chain(acge_retriever_chain, document_chain)
acge_results = []
for question in questions:
    result = invoke_question(acge_qa, question)
    if result:
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
        acge_results.append({"question": question, "answer": result['answer']})
    else:
        print(f"-> **Question**: {question} \n")
        print("**Answer**: 无法获取答案，发生错误。\n")
        acge_results.append({"question": question, "answer": "无法获取答案，发生错误。"})




# 初始化 bm25 检索器
bm25_retriever = BM25Retriever.from_documents(document_list)
# First we need a prompt that we can pass into an LLM to generate this search query
bm25_retriever_chain = create_history_aware_retriever(llm, bm25_retriever, prompt)
bm25_qa = create_retrieval_chain(bm25_retriever_chain, document_chain)
bm25_results = []
for question in questions:
    result = invoke_question(bm25_qa, question)
    if result:
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
        bm25_results.append({"question": question, "answer": result['answer']})
    else:
        print(f"-> **Question**: {question} \n")
        print("**Answer**: 无法获取答案，发生错误。\n")
        bm25_results.append({"question": question, "answer": "无法获取答案，发生错误。"})




# 初始化bge检索器
bge_model_path = get_env_variable("Local_Model_bge_text_embedding")
# 初始化 LocalEmbeddings 实例，使用本地模型的路径
local_embeddings_bge = LocalEmbeddings(
    model_path=bge_model_path,  # 本地模型文件的路径
)
# 初始化一个空的 Chroma 数据库
bge_db = Chroma(embedding_function=local_embeddings_bge, collection_name="bge_collection")
# 设置批处理大小
batch_size = 100  # 批处理大小
batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
total_batches = len(batches)
print(f"总批次 {total_batches}")
# 并行计算嵌入并合并结果
with ThreadPoolExecutor(max_workers=cpu_count) as executor, tqdm(total=total_batches, desc="Computing embeddings") as progress_bar:
    futures = [executor.submit(compute_embeddings, bge_db, batch, i, total_batches, progress_bar) for i, batch in enumerate(batches)]
    for future in as_completed(futures):
        future.result()
print("Document database created, bge_db=", bge_db)
# 获取索引中的文档数量
num_documents_in_index = len(bge_db._collection.get()["documents"])
print(f"索引中的文档数量:{num_documents_in_index}")
# 创建检索器，可以使用 'mmr' 或 'similarity' 检索类型
bge_retriever = bge_db.as_retriever(
    search_type="mmr",  # 或 "similarity"
    search_kwargs={"k": 8, 'fetch_k': min(20, num_documents_in_index)},  # k: Amount of documents to return (Default: 4); fetch_k: Amount of documents to pass to MMR algorithm
)
bge_retriever_chain = create_history_aware_retriever(llm, bge_retriever, prompt)
bge_qa = create_retrieval_chain(bge_retriever_chain, document_chain)
bge_results = []
for question in questions:
    result = invoke_question(bge_qa, question)
    if result:
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")
        bge_results.append({"question": question, "answer": result['answer']})
    else:
        print(f"-> **Question**: {question} \n")
        print("**Answer**: 无法获取答案，发生错误。\n")
        bge_results.append({"question": question, "answer": "无法获取答案，发生错误。"})





# 创建结果存储目录
results_dir = "./result"
os.makedirs(results_dir, exist_ok=True) 
with open(os.path.join(results_dir, "acge_results.json"), "w", encoding='utf-8') as f:
    json.dump(acge_results, f, ensure_ascii=False, indent=4)
with open(os.path.join(results_dir, "bm25_results.json"), "w", encoding='utf-8') as f:
    json.dump(bm25_results, f, ensure_ascii=False, indent=4)
with open(os.path.join(results_dir, "bge_results.json"), "w", encoding='utf-8') as f:
    json.dump(bge_results, f, ensure_ascii=False, indent=4)
