'''
基于langchain+local model（llama-2-13b.gguf.q4_0.bin）搭建简单的RAG系统
'''

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
import torch


# Load data.
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
print(all_splits)


# Download GPT4All embeddings locally.
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(device=device))

# Test similarity search is working with our local embeddings.
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(len(docs))
print(docs[0])

# Set model

# 这个需要从网上下载模型
# model = GPT4All("D:/Project/GPT4All/models/orca-mini-3b-gguf2-q4_0.gguf")
# output = model.generate("The capital of France is ", max_tokens=3)
# print(output)

# Load local model：GTP4ALL
# from gpt4all import GPT4All
# model_path = "D:/Project/GPT4All/models/"
# model_name = "Nous-Hermes-13B.Q4_0.gguf.bin"
# model = GPT4All(model_name=model_name, model_path=model_path)
# tokens = []
# with model.chat_session():
#     for token in model.generate("What is the capital of France?", streaming=True):
#         tokens.append(token)
# print(tokens)

# Load local model：llama2
llm = LlamaCpp(
    model_path="./Llama-2-13B-GGML/llama-2-13b.gguf.q4_0.bin", # Make sure the model path is correct for your system!
    n_gpu_layers=1, # Metal set to 1 is enough.
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
    device=device
)
# llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")


# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result=rag_chain.invoke("What is Task Decomposition?")
print(result)
# cleanup
vectorstore.delete_collection()