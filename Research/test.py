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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import random
import math

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
# repo_path = get_env_variable("Repo_Path")+ "/libs/core/langchain_core" # 测试用的
repo_path = get_env_variable("Python_Path")

# 获取所有子文件夹
subfolders = [os.path.join(repo_path, f) for f in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, f))]
total_subfolders = len(subfolders)
print(f"总共有 {total_subfolders} 个子文件夹。")

# 计算需要随机选取的文件夹数量，向上取整
num_to_select = math.ceil(total_subfolders * 0.4)
selected_subfolders = random.sample(subfolders, num_to_select)
print(f"随机选取了 {num_to_select} 个子文件夹进行读取。")

# 创建 GenericLoader 实例，并显示进度条
for folder in selected_subfolders:
    print(f"正在处理文件夹: {folder}")
    loader = GenericLoader.from_filesystem(
        path=folder,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),  # parser_threshold: Minimum lines needed to activate parsing (0 by default).
        show_progress=True  # 显示进度条
    )
    documents = list(loader.lazy_load())
    print(f"文件夹 {folder} 中加载了 {len(documents)} 个文档。")

# loader = GenericLoader.from_filesystem(
#     path = repo_path,
#     glob="**/*",
#     suffixes=[".py"],
#     exclude=["**/non-utf8-encoding.py"],
#     parser=LanguageParser(language=Language.PYTHON, parser_threshold=500), # parser_threshold: Minimum lines needed to activate parsing (0 by default).
#     show_progress=True  # 显示进度条
# )

# 获取文档生成器并展开
document_list = list(loader.lazy_load())
print(f"len(document_list): {len(document_list)}")


# 并行加载文档
documents = []

# 打印 CPU 核心数
cpu_count = os.cpu_count()
print(f"系统中可用的 CPU 核心数: {cpu_count}")
with ThreadPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(lambda doc: doc, doc) for doc in document_list]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading documents"):
        documents.append(future.result())

