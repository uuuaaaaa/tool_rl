from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

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

def call_gpt(prompt):
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

# 示例使用
if __name__ == "__main__":
    result = call_gpt("你好，请介绍一下你自己。")
    print(result)


# from transformers import pipeline
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import ray

# # 检查是否在Ray环境中运行
# if ray.is_initialized():
#     print("Running inside Ray cluster")
#     # 获取Ray分配的GPU
#     gpu_ids = ray.get_gpu_ids()
#     if gpu_ids:
#         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
#         print(f"Ray assigned GPUs: {gpu_ids}")
# else:
#     print("Running outside Ray cluster")
#     # 使用所有可用GPU
#     if torch.cuda.is_available():
#         gpu_count = torch.cuda.device_count()
#         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))

# # 全局变量，确保模型只加载一次
# _model = None
# _tokenizer = None
# _pipeline = None

# def initialize_model():
#     """初始化模型，确保只加载一次"""
#     global _model, _tokenizer, _pipeline
    
#     if _model is not None:
#         return _model, _tokenizer, _pipeline
    
#     model_name = "/run/determined/NAS1/public/HuggingFace/Qwen/Qwen3-1.7B"
    
#     print("Loading tokenizer...")
#     _tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         local_files_only=True
#     )
    
#     print("Loading model...")
#     # 根据环境选择设备映射策略
#     if ray.is_initialized() and ray.get_gpu_ids():
#         # 在Ray环境中，使用更明确的设备映射
#         _model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,  # 使用半精度节省内存
#             low_cpu_mem_usage=True,
#             device_map="auto",  # 自动分配到所有可用GPU
#         )
#     else:
#         # 非Ray环境
#         _model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto",  # 自动分配到所有可用GPU
#             low_cpu_mem_usage=True,
#         )
    
#     print("Creating pipeline...")
#     _pipeline = pipeline(
#         task="text-generation", 
#         model=_model, 
#         tokenizer=_tokenizer
#     )
    
#     print(f"Model loaded on device: {next(_model.parameters()).device}")
#     return _model, _tokenizer, _pipeline

# def call_gpt(prompt):
#     """调用GPT模型生成文本"""
#     global _model, _tokenizer, _pipeline
    
#     # 确保模型已初始化
#     if _model is None:
#         _model, _tokenizer, _pipeline = initialize_model()
    
#     print(f"Model is on device: {next(_model.parameters()).device}")
    
#     # 使用预先创建的pipeline
#     output = _pipeline(prompt, max_new_tokens=100)[0]["generated_text"]
#     return output