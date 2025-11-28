import json
import os
from search_r1.llm_agent.image_detection import diffusion_generated_detection, manipulation_detection

# # 读取 JSON 文件
# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# # 保存结果到新的 JSON 文件
# def save_results_to_json(results, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(results, file, indent=4)

# # 主函数
# def main():
#     input_file_path = '/home/chenhui/EasyR1/val.json'
#     output_file_path = '/home/chenhui/EasyR1/detection_results.json'
    
#     # 读取 JSON 文件内容
#     data = read_json_file(input_file_path)
    
#     # 提取所有图像路径
#     all_images = [item['images'][0] for item in data]
    
#     # 组合结果
#     results = []
#     for image_path in all_images:
#         # 调用检测函数
#         # diffusion_result = diffusion_generated_detection(image_path)
#         manipulation_result = manipulation_detection(image_path)
        
#         # 组合结果
#         combined_result = {
#             image_path: {
#                 # "diffusion_generated": diffusion_result,
#                 "manipulated": manipulation_result
#             }
#         }
#         results.append(combined_result)
    
#     # 将结果保存到新的 JSON 文件
#     save_results_to_json(results, output_file_path)
    
#     print(f"Results saved to {output_file_path}")


# # 读取 JSON 文件
# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# # 保存结果到新的 JSON 文件
# def save_results_to_json(results, file_path):
#     with open(file_path, 'w') as file:
#         json.dump(results, file, indent=4)

# # 主函数
# def main():
#     input_file_path = '/home/chenhui/EasyR1/DGM4_train_5000.json'
#     output_file_path = '/home/chenhui/EasyR1/detection_results.json'
    
#     # 读取原始 JSON 文件内容
#     data = read_json_file(input_file_path)
    
#     # 提取所有图像路径
#     all_images = [item['images'][0] for item in data]
    
#     # 检查输出文件是否存在
#     existing_results = []
#     start_index = 0
    
#     if os.path.exists(output_file_path):
#         try:
#             existing_results = read_json_file(output_file_path)
#             # 计算已处理的数量，从下一个开始
#             start_index = len(existing_results)
#             print(f"检测到已有 {start_index} 条结果，从第 {start_index + 1} 条开始继续检测")
#         except (json.JSONDecodeError, Exception) as e:
#             print(f"读取现有结果文件失败，将从头开始检测: {e}")
#             existing_results = []
#             start_index = 0
    
#     # 如果已经处理完所有图像，直接返回
#     if start_index >= len(all_images):
#         print("所有图像已处理完成！")
#         return
    
#     # 使用现有结果或创建新的结果列表
#     results = existing_results
    
#     # 从上次保存的位置继续检测
#     for i in range(start_index, len(all_images)):
#         image_path = all_images[i]
        
#         try:
#             # 调用检测函数
#             diffusion_result = diffusion_generated_detection(image_path)
#             manipulation_result = manipulation_detection(image_path)
            
#             # 组合结果
#             combined_result = {
#                 image_path: {
#                     "diffusion_generated": diffusion_result,
#                     "manipulated": manipulation_result
#                 }
#             }
#             results.append(combined_result)
            
#             # 每处理一条就保存一次
#             save_results_to_json(results, output_file_path)
#             print(f"已处理第 {i + 1}/{len(all_images)} 条: {image_path}")
            
#         except Exception as e:
#             print(f"处理第 {i + 1} 条时出错 ({image_path}): {e}")
#             # 即使出错也保存当前进度
#             save_results_to_json(results, output_file_path)
#             print(f"已保存当前进度，共 {len(results)} 条结果")
    
#     print(f"所有处理完成！最终结果保存到 {output_file_path}")

def extract_detection_results(image_path):
    """
    从指定的JSON文件中提取特定图像路径的检测结果。
    
    :param image_path: 要提取结果的图像路径
    :param json_file_path: 保存检测结果的JSON文件路径
    :return: 包含"diffusion_generated"和"manipulated"的字典，如果未找到则返回None
    """
    # 读取JSON文件
    json_file_path='/home/chenhui/EasyR1/detection_results.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        if image_path in item:
            diffusion_generated_str = item[image_path]['diffusion_generated']
            manipulated_str = item[image_path]['manipulated']
            return "The image is "+diffusion_generated_str + " and "+manipulated_str
    # 如果未找到匹配的图像路径，返回None
    return None

def manipulation_detection(image_path):
    json_file_path='/home/chenhui/EasyR1/detection_results_MMFakeBench.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        if image_path in item:
            manipulated_str = item[image_path]['manipulated']
            return "The image is "+manipulated_str
    # 如果未找到匹配的图像路径，返回None
    return None

def diffusion_detection(image_path):
    json_file_path='/home/chenhui/EasyR1/detection_results_MMFakeBench.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        if image_path in item:
            diffusion_generated_str = item[image_path]['diffusion_generated']
            return "The image is "+diffusion_generated_str
    # 如果未找到匹配的图像路径，返回None
    return None
def image_detection(image_path):
    json_file_path='/home/chenhui/EasyR1/detection_results_AMG.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        if image_path in item:
            image_str = item[image_path]['prediction']
            return "The image is "+image_str
    # 如果未找到匹配的图像路径，返回None
    return None

if __name__ == "__main__":
    main()