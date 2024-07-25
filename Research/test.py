import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from dotenv import load_dotenv
import os


# 加载.env文件中的环境变量
load_dotenv() 
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

# 加载结果文件
acge_results = json.load(open("./result/acge_results.json", encoding='utf-8'))
bge_results = json.load(open("./result/bge_results.json", encoding='utf-8'))
bm25_results = json.load(open("./result/bm25_results.json", encoding='utf-8'))

# 加载排序模型
model_path = get_env_variable("Local_Model_bge_reranker")
tokenizer = AutoTokenizer.from_pretrained(model_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(model_path)
rerank_model.cuda()
rerank_model.eval()

fusion_results = []
k = 60

for q_acge, q_bge, q_bm25 in zip(acge_results, bge_results, bm25_results):
    fusion_score = {}

    # 合并ACGE和BGE的结果
    for idx, doc in enumerate(q_acge['reference']):
        if doc not in fusion_score:
            fusion_score[doc] = 1 / (idx + k)
        else:
            fusion_score[doc] += 1 / (idx + k)
    
    for idx, doc in enumerate(q_bge['reference']):
        if doc not in fusion_score:
            fusion_score[doc] = 1 / (idx + k)
        else:
            fusion_score[doc] += 1 / (idx + k)

    # 合并BM25的结果
    for idx, doc in enumerate(q_bm25['reference']):
        if doc not in fusion_score:
            fusion_score[doc] = 1 / (idx + k)
        else:
            fusion_score[doc] += 1 / (idx + k)

    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    # 重排序
    pairs = []
    for sorted_result in sorted_dict[:3]:
        pairs.append([q_acge["question"], sorted_result[0]])

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    sorted_result = sorted_dict[scores.cpu().numpy().argmax()]
    q_acge['reference'] = sorted_result[0]
    fusion_results.append(q_acge)

# 保存融合和重排序后的结果
with open('./result/submit_fusion_bge+bm25_rerank_retrieval.json', 'w', encoding='utf8') as f:
    json.dump(fusion_results, f, ensure_ascii=False, indent=4)
