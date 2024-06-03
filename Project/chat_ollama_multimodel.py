import requests
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志配置，日志将保存到文件中
logging.basicConfig(filename='./Project/log/model_responses.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API URL
url_generate = "http://localhost:11434/api/generate"

# Function to get response from the model
def get_response(url, model_name, data):
    logger.info(f"Sending request to model: {model_name} with prompt: {data['prompt']}")
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Ensure the request was successful
        response_dict = response.json()
        response_content = response_dict.get("response")
        if response_content is None:
            error_msg = f"Error: 'response' field not found. Full response: {response_dict}"
            logger.error(error_msg)
            return {"model": model_name, "response": error_msg}
        logger.info(f"Received response from model: {model_name}")
        return {"model": model_name, "response": response_content}
    except requests.RequestException as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return {"model": model_name, "response": error_msg}
    except json.JSONDecodeError:
        error_msg = "Error: Failed to decode JSON response from server"
        logger.error(error_msg)
        return {"model": model_name, "response": error_msg}

# List of model request data
data_list = [
    {
        "model": "gemma:2b",
        "prompt": "Why is the sky blue?",
        "stream": False
    },
    {
        "model": "qwen",
        "prompt": "Why is the sky blue?",
        "stream": False
    }
]

# Use ThreadPoolExecutor to make concurrent requests
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_response, url_generate, data["model"], data) for data in data_list]

    for future in as_completed(futures):
        try:
            result = future.result()
            model = result['model']
            response = result['response']
            output = f"\n{'='*40}\nModel: {model}\nResponse:\n{response}\n{'='*40}\n"
            print(output)
            logger.info(output)
        except Exception as e:
            error_msg = f"Request generated an exception: {e}"
            print(f"\n{'='*40}\n{error_msg}\n{'='*40}\n")
            logger.error(error_msg)
