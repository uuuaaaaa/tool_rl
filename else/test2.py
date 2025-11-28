# import json

# def convert_dict_to_list(input_file, output_file):
#     """
#     将JSON文件从字典格式转换为列表格式
    
#     Args:
#         input_file: 输入文件路径
#         output_file: 输出文件路径
#     """
#     # 读取原始JSON文件
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data_dict = json.load(f)
    
#     # 将字典的值转换为列表
#     data_list = list(data_dict.values())
    
#     # 保存为新的JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data_list, f, ensure_ascii=False, indent=2)
    
#     print(f"转换完成！")
#     print(f"原始数据条数: {len(data_dict)}")
#     print(f"转换后数据条数: {len(data_list)}")
#     print(f"输出文件: {output_file}")

# # 使用示例
# if __name__ == "__main__":
#     input_file = "/home/chenhui/EasyR1/MR2/dataset_items_test.json"
#     output_file = "/home/chenhui/EasyR1/MR2/dataset_items_test_list.json"
    
#     convert_dict_to_list(input_file, output_file)
    
#     # 验证转换结果
#     with open(output_file, 'r', encoding='utf-8') as f:
#         result = json.load(f)
#         print(f"\n前3条数据示例:")
#         for i, item in enumerate(result[:3]):
#             print(f"索引 {i}: {item}")
# import json
# import random
# import os

# def extract_random_entries_advanced(input_file, output_file, num_entries=1500):
#     """
#     从JSON文件中随机提取指定数量的条目（增强版）
#     """
    
#     try:
#         # 检查输入文件是否存在
#         if not os.path.exists(input_file):
#             print(f"错误: 输入文件 {input_file} 不存在")
#             return False
        
#         # 读取原始文件
#         print(f"正在读取文件: {input_file}")
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         print(f"✓ 成功读取文件，共有 {len(data)} 条数据")
        
#         # 检查数据格式
#         if not isinstance(data, list):
#             print("错误: JSON文件格式不正确，应该是一个列表")
#             return False
        
#         # 检查请求的数量是否超过总数据量
#         if num_entries > len(data):
#             print(f"警告: 请求的条目数({num_entries})超过总数据量({len(data)})")
#             print(f"将提取所有 {len(data)} 条数据")
#             num_entries = len(data)
        
#         # 随机抽取指定数量的条目
#         print(f"正在随机抽取 {num_entries} 条数据...")
#         random_entries = random.sample(data, num_entries)
        
#         # 确保输出目录存在
#         output_dir = os.path.dirname(output_file)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         # 保存到新文件
#         print(f"正在保存到: {output_file}")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(random_entries, f, ensure_ascii=False, indent=4)
        
#         print(f"✓ 成功提取 {len(random_entries)} 条数据")
#         print(f"✓ 文件已保存到: {output_file}")
        
#         return True
        
#     except Exception as e:
#         print(f"错误: {str(e)}")
#         return False

# # 使用示例
# if __name__ == "__main__":
#     input_path = "/home/chenhui/EasyR1/MR2/MR2_trian.json"
#     output_path = "/home/chenhui/EasyR1/MR2_data/MR2_1000.json"
    
#     success = extract_random_entries_advanced(input_path, output_path, 1000)
    
#     if success:
#         print("\n操作完成！")
#     else:
#         print("\n操作失败！")



import json
import os

# 文件路径设置
input_file = "/home/chenhui/EasyR1/MR2_data/MR2_1000.json"
output_file = "/home/chenhui/EasyR1/MR2_data/train_1000grpo.json"
base_path = "/home/chenhui/EasyR1/MR2"

# input_file = "/home/chenhui/EasyR1/MR2/dataset_items_test_list.json"
# output_file = "/home/chenhui/EasyR1/MR2_data/val_grpo.json"
# base_path = "/home/chenhui/EasyR1/MR2"

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换数据
converted_data = []
prompt_template = """<image>The news caption is: {text}
Given a news caption and an accompanying image, your task is to classify the type of misinformation in this social media post.\n
The three possible categories are:\n
- Non_Rumor: A post that is verified to be true or mostly accurate.\n
- Rumor: A post that is verified to be false, misleading, or manipulated.\n
- Unverified: A post that cannot be confirmed as true or false due to lack of evidence.\n

Please follow these steps:\n
1.Reason step by step through an internal monologue, enclosed within <think> </think> tags.Check for potential mismatches between visual content and textual claims.And Look for signs of image editing or contextual manipulation.\n
2.Based on the above information,provide your answer in the format: <answer> type of misinformation</answer>,for example: <answer>Non_Rumor</answer>.\n
 """
# 3.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> original </answer>.\n

# 2.After reasoning,you can call the search engine when need in the form: <search>{text}</search> to detect factual errors in the news caption.\n
# 3.After reasoning,you can also call the tool when need in the form:<tool> manipulation_detection </tool> to detect manipulation in the image,or call the tool in the form :<tool> diffusion_detection </tool> to detect diffusion in the image.\n
for item in data:
    converted_item = {
        "images": [os.path.join(base_path, item["image_path"].lstrip('/'))],
        "problem": prompt_template.format(text=item["caption"]),
        "answer": item["label"]
    }
    converted_data.append(converted_item)

# 保存转换后的数据
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)

print(f"转换完成！共处理 {len(converted_data)} 条数据")
