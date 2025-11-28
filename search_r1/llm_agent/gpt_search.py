import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from tqdm import tqdm
# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 全局变量，避免重复加载
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    
    # model_name = "/run/determined/NAS1/public/HuggingFace/Qwen/Qwen3-32B/qwen3_32b/"
    model_name = "/run/determined/NAS1/public/HuggingFace/Qwen/Qwen3-8B"
    # 配置4位量化以节省内存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # 添加pad_token如果不存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配设备
        torch_dtype=torch.float16,  # 使用半精度
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # 设置为评估模式
    model.eval()

def call_local_model(prompt):
    global model, tokenizer
    
    # 延迟初始化
    if model is None or tokenizer is None:
        init_model()
    
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    print("finish")
    # print(generated_text)
    return generated_text

def call_zhipu_api(messages, model_name="glm-4.5"):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    headers = {
        "Authorization": "e8c7bbc6ea5b4f63827099066fad12b5.7qH1sNV3wDMXiKqC",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.6
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            # 检查是否是身份验证失败
            error_data = response.json()
            if "身份验证失败" in str(error_data):
                raise Exception("API密钥无效或身份验证失败")
            raise Exception(f"API调用失败: {response.status_code}, {response.text}")
            
    except Exception as e:
        print(f"API调用异常: {e}")
        raise e


def call_gpt(prompt):
    # 首先尝试调用API
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]
        result = call_zhipu_api(messages)
        result = result['choices'][0]['message']['content']
        print("API调用成功")
        return result
        
    except Exception as e:
        print(f"API调用失败，尝试使用本地模型: {e}")
        # API调用失败时使用本地模型
        return call_local_model(prompt)


# 示例使用
if __name__ == "__main__":
    # 初始化模型

    
    # 读取JSON文件
    input_file = "/home/chenhui/EasyR1/MMFakeBench_test/source/train.json"
    output_file = "search_results.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果输出文件已存在，读取已有结果
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except:
            existing_results = []
    
    # 处理每个条目
    for i, item in enumerate(tqdm(data, desc="Processing items")):
        # 跳过已经处理过的条目
        if i < len(existing_results):
            continue
            
        text = item["text"]
        image_path = item["image_path"]
        
        # 构建prompt
        prompt = f"The news caption is {text}, please determine if there are any factual errors in the news caption. If so, directly answer in the form '<answer>False</answer>'. If not, directly answer answer in the form:'<answer>True</answer>'"
        
        # 调用模型
        try:
            output = call_gpt(prompt)
            
            # 构建结果项
            result_item = {
                image_path: {
                    "output": output
                }
            }
            print(output)
            # 添加到结果列表
            if i < len(existing_results):
                existing_results[i] = result_item
            else:
                existing_results.append(result_item)
            
            # 立即保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            
            print(f"Processed item {i+1}/{len(data)}: {image_path}")
            
        except Exception as e:
            print(f"Error processing item {i+1} with image_path {image_path}: {e}")
            # 即使出错也添加一个错误结果
            result_item = {
                image_path: {
                    "output": f"Error: {str(e)}"
                }
            }
            
            if i < len(existing_results):
                existing_results[i] = result_item
            else:
                existing_results.append(result_item)
            
            # 保存错误结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
    
    print(f"Processing completed. Results saved to {output_file}")