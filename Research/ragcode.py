from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from typing import List

# Clone
repo_path = "D:/Project/langchain"
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(len(texts))

# RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from LocalEmbedding import LocalEmbeddings

# 初始化 LocalEmbeddings 实例
local_embeddings = LocalEmbeddings(
    model_name='bert-base-uncased'  # 使用 BERT 模型
)

# 使用 LocalEmbeddings 和 Chroma 创建文档数据库
db = Chroma.from_documents(texts, local_embeddings)
print("db=",db)
# 创建检索器，可以使用 'mmr' 或 'similarity' 检索类型
retriever = db.as_retriever(
    search_type="mmr",  # 或 "similarity"
    search_kwargs={"k": 8}  # k 表示检索的文档数量
)

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

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
print(result["answer"])





