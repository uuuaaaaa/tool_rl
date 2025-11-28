import json
def extract_search_results(image_path):
    """
    从指定的JSON文件中提取特定图像路径的检测结果。
    
    :param image_path: 要提取结果的图像路径
    :param json_file_path: 保存检测结果的JSON文件路径
    :return: 包含"diffusion_generated"和"manipulated"的字典，如果未找到则返回None
    """
    # 读取JSON文件
    json_file_path='/home/chenhui/EasyR1/search_results.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        if image_path in item:
            search_output = item[image_path]['output']
            return  search_output
    # 如果未找到匹配的图像路径，返回None
    return None

def create_output_json():
    # 读取原始JSON文件
    input_file = "/home/chenhui/EasyR1/train_1000tool3.json"
    output_file = "output_results.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    # 处理每个条目
    for i, item in enumerate(data):
        image_path = item["images"][0]  # 获取images数组的第一个元素
        
        # 根据索引范围确定output内容
        if 600 <= i < 900:  # 第601-900条（索引从0开始，所以600-899）
            output = "there are factual errors in the news caption"
        else:  # 第1-600条和901-1000条
            output = "there are not any factual errors in the news caption"
        
        # 构建结果项
        result_item = {
            image_path: {
                "output": output
            }
        }
        results.append(result_item)
    
    # 保存结果到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理了 {len(data)} 条数据")
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    create_output_json()