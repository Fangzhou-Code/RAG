# 介绍各个文件

* 基于langchain+local model（llama-2-13b.gguf.q4_0.bin）搭建简单的RAG系统： [localrum.py](./localrun.py)
* RAG 完整流程和基于图片的RAG系统搭建: [demo.ipynb](./demo.ipynb)

RAG流程图：

![RAG流程图](./image/rag1.jpeg)

![RAG流程图](./image/rag2.jpeg)


知识库流程图：

![知识库流程图](./image/kg.png)



## Ollama

### 资料

* [Ollama 可以在 Windows 上运行了](https://blog.csdn.net/engchina/article/details/136125933)
* [Ollama 支持同时加载多个模型,单个模型同时处理多个请求 ](https://www.bilibili.com/read/cv34357822/)
* [xinference + dify + ollama 构建本地知识库](https://mp.weixin.qq.com/s/XrHZqXZ-8oV2kKOgfUlcIw)
* [FastGPT + OneAPI + xinferencce + ollama 构建本地知识库](https://www.53ai.com/news/qianyanjishu/1260.html)
* [dify+ollama构建本地大模型平台](https://zhuanlan.zhihu.com/p/697386670)

### dify+ollama

**环境搭建**

* wsl2 + docker
* 关闭防火墙


**启动流程**

* 首先启动ollama: `ollama run MODEL_NAME`
* 本地clone dify仓库：`git clone https://github.com/langgenius/dify.git`
* 进入下载后的文件夹中的docker文件夹: `cd dify/docker`
  
  ![dify 结构](/image/dify.png "DiFy 目录结构")
* 启动docker: `docker compose up -d`
  
  第一次启动，因为要下载Images，需要等一段时间。启动后查看Docker Desktop的界面：
  ![Docker Desktop的界面](/image/doker-desktop.png "Docker Desktop的界面")
  
  如果需要修改配置，可以参考:https://docs.dify.ai/v/zh-hans/getting-started/install-self-hosted/environments 修改docker-compose.yaml文件。

* 启动成功后访问127.0.0.1
  
  登陆邮箱： 945183225@qq.com

  密码: root1234
  
  账户名： root

  ![Dify本地界面](/image/dify-ui.png "Dify本地界面")

* 绑定Ollama首先本地启用，嵌入选择url：`http://host.docker.internal:11434`
* 后续操作参考: [dify+ollama构建本地大模型平台](https://zhuanlan.zhihu.com/p/697386670)


### ollama调用多模型

* 首先本地启动`ollama serve`
* 运行 `Project/chat_ollama_multimodel.py`



## 多模型平台

* fastchat: [FastChat](https://github.com/lm-sys/FastChat?tab=readme-ov-file#serving-with-web-gui)
  * [使用 FastChat 快速部署 LLM 服务 + VLLM](https://rudeigerc.dev/posts/llm-inference-with-fastchat/)
  * [使用 FastChat 部署 LLM](https://zhaozhiming.github.io/2023/08/22/use-fastchat-deploy-llm/)
  * [FastChat 框架中的服务解析](http://felixzhao.cn/Articles/article/71)
* chatall：[齐叨](https://github.com/sunner/ChatALL)
* ChatHub: [ChatHub](https://chathub.gg/)


### FastChat 部署多模型流程

* 下载FastChat: </br>
`git clone https://github.com/lm-sys/FastChat.git`</br>
`cd FastChat`</br>
if you runnning on MAC:`brew install rust cmake`
* 安装包：</br>
`pip3 install --upgrade pip  # enable PEP 660 support`</br>
`pip3 install -e ".[model_worker,webui]"`
* 下载两个模型：</br>
  `git lfs install`</br>
  `git clone https://huggingface.co/lmsys/vicuna-7b-v1.5`</br>
  `git clone https://huggingface.co/lmsys/longchat-7b-v1.5-32k`
* （可选）终端交互试下能否运行：`python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5`
* 启动控制器服务：`python -m fastchat.serve.controller --host 0.0.0.0`
* 启动Worker服务：</br>
第一个模型：CUDA_VISIBLE_DEVICES=0</br>
`
 python -m fastchat.serve.model_worker --model-path ../vicuna-7b-v1.5 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
`</br>
第二个模型：CUDA_VISIBLE_DEVICES=1</br>
`
 python -m fastchat.serve.model_worker --model-path ../longchat-7b-v1.5-32k  --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
`
* 启动 RESTFul API 服务：`
python -m fastchat.serve.openai_api_server --host 0.0.0.0
`</br>
可以通过访问`http://127.0.0.1:8000/docs`可以查看接口信息
![FastAPI](/image/FastAPI.png)
* WebUI部署：`
python -m fastchat.serve.gradio_web_server_multi
`</br>
访问`127.0.0.1:7860`访问UI界面，选择`side-by-side`下图红色方框
![FastChat-UI](/image/FastChat-UI.png)