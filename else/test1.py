# import json
# import os
# import glob

# # 文件路径设置
# input_file = "/home/chenhui/EasyR1/AMG_data/AMG_2000.json"
# base_path = "/home/chenhui/EasyR1/AMG/AMG_MEDIA/train"
# output_file = "/home/chenhui/EasyR1/AMG_data/AMG_train_tool3.json"
# # input_file = "/home/chenhui/EasyR1/AMG-An-Attributing-Multi-modal-Fake-News-Dataset/dataset/test.json"
# # base_path = "/home/chenhui/EasyR1/AMG/AMG_MEDIA/test"
# # output_file = "/home/chenhui/EasyR1/AMG_test_tool3.json"

# def find_media_files(base_path, file_id):
#     """
#     在指定路径下查找对应ID的媒体文件（图片或视频）
#     """
#     # 可能的图片扩展名
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
#     # 可能的视频扩展名  
#     video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
#     all_extensions = image_extensions + video_extensions
    
#     # 查找所有可能的文件
#     for ext in all_extensions:
#         pattern = os.path.join(base_path, f"{file_id}{ext}")
#         matches = glob.glob(pattern)
#         if matches:
#             return matches[0]  # 返回第一个匹配的文件
    
#     return None
# # 转换数据
# converted_data = []
# prompt_template = """<image>The news caption is: {text}
# Given a news caption and an accompanying image, your task is to classify the type of multimodal misinformation.\n
# The six possible categories are:\n
# - Image Fabrication: The image is forged or manipulated.\n
# - Entity Inconsistency：There is a discrepancy between the key entities depicted in the textual and visual modalities.\n
# - Event Inconsistency：There is misrepresentation that arises from excessive inference in the written text for the attached image.\n
# - Time & Space Inconsistency：Using unaltered images or videos that depict past events, but falsely presenting them as recent events.\n
# - Ineffective Visual Information: The image consists of textual information  cannot provide evidence or proof for news content.\n
# - Real News: No form of multimodal misinformation is detected.\n

# Please follow these steps:\n
# 1.Reason step by step through an internal monologue, enclosed within <think> </think> tags.\n
# 2.After reasoning,you can call the tool when need in the form:<tool> forge_detection </tool> to detect image fabrication in the image or call the tool in the form:<tool> manipulation_detection </tool> to detect manipulation in the image.\n
# 3.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> Real News </answer>.\n
#  """
# # 2.After reasoning,you can also call the tool when need in the form:<tool> manipulation_detection </tool> to detect manipulation in the image,or call the tool in the form :<tool> diffusion_detection </tool> to detect diffusion in the image.\n
# # 2.After reasoning,you can call the search engine when need in the form: <search>{text}</search> to detect factual errors in the news caption.\n
# # 2. If needed, you may use **one** of the following tools during your reasoning:
# #    - Use `<search>{text}</search>` to verify factual claims in the news caption.
# #    - Use `<tool>manipulation_detection</tool>` to check for image manipulation.
# #    - Use `<tool>diffusion_detection</tool>` to detect if the image is AI-generated.
# # 3.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> original </answer>.\n
# # 2.After reasoning,you can also call the tool when need in the form:<tool> manipulation_detection </tool> to detect manipulation in the image,or call the tool in the form :<tool> diffusion_detection </tool> to detect diffusion in the image.\n
# #After reasoning,you can also call the tool when need in the form:<tool> image_detection </tool> to detect image fabrication in the image.\n

# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# for item in data:
#     item_id = item['Id']
#     media_path = find_media_files(base_path, item_id)
#     converted_item = {
#         "images": [media_path],
#         "problem": prompt_template.format(text=item["content"]),
#         "answer": item["label"]
#     }
#     converted_data.append(converted_item)

# # 保存转换后的数据
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(converted_data, f, indent=4, ensure_ascii=False)

# print(f"转换完成！共处理 {len(converted_data)} 条数据")

# import json
# import os
# import glob

# # 文件路径设置
# input_file = "/home/chenhui/EasyR1/AMG-An-Attributing-Multi-modal-Fake-News-Dataset/dataset/train.json"
# base_path = "/home/chenhui/EasyR1/AMG/AMG_MEDIA/train"
# output_file = "/home/chenhui/EasyR1/AMG_train_tool1.json"
# # input_file = "/home/chenhui/EasyR1/AMG-An-Attributing-Multi-modal-Fake-News-Dataset/dataset/test.json"
# # base_path = "/home/chenhui/EasyR1/AMG/AMG_MEDIA/test"
# # output_file = "/home/chenhui/EasyR1/AMG_test_tool1.json"

# def find_media_files(base_path, file_id):
#     """
#     在指定路径下查找对应ID的媒体文件（图片或视频）
#     """
#     # 可能的图片扩展名
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
#     # 可能的视频扩展名  
#     video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    
#     all_extensions = image_extensions + video_extensions
    
#     # 查找所有可能的文件
#     for ext in all_extensions:
#         pattern = os.path.join(base_path, f"{file_id}{ext}")
#         matches = glob.glob(pattern)
#         if matches:
#             return matches[0]  # 返回第一个匹配的文件
    
#     return None

# def get_media_type(file_path):
#     """
#     根据文件扩展名判断媒体类型
#     """
#     if not file_path:
#         return "unknown"
    
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
#     video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
#     _, ext = os.path.splitext(file_path)
#     ext = ext.lower()
    
#     if ext in image_extensions:
#         return "image"
#     elif ext in video_extensions:
#         return "video"
#     else:
#         return "unknown"

# # 转换数据
# converted_data = []

# # 图片和视频的提示模板
# image_prompt_template = """<image>The news caption is: {text}
# Given a news caption and an accompanying image, your task is to classify the type of multimodal misinformation.\n
# The six possible categories are:\n
# - Image Fabrication: The image is AI-generated or manipulated.\n
# - Entity Inconsistency：There is a discrepancy between the key entities depicted in the textual and visual modalities.\n
# - Event Inconsistency：There is misrepresentation that arises from excessive inference in the written text for the attached image.\n
# - Time & Space Inconsistency：Using unaltered images that depict past events, but falsely presenting them as recent events.\n
# - Ineffective Visual Information: The image consists of textual information  cannot provide evidence or proof for news content.\n
# - Real News: No form of multimodal misinformation is detected.\n

# Please follow these steps:\n
# 1.Reason step by step through an internal monologue, enclosed within <think> </think> tags.\n
# 2.After reasoning,you can also call the tool when need in the form:<tool> manipulation_detection </tool> to detect manipulation in the image,or call the tool in the form :<tool> diffusion_detection </tool> to detect diffusion in the image.\n
# 3.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> Real News </answer>.\n
# """

# video_prompt_template = """<video>The news caption is: {text}
# Given a news caption and an accompanying video, your task is to classify the type of multimodal misinformation.\n
# The six possible categories are:\n
# - Image Fabrication: The video is AI-generated or manipulated.\n
# - Entity Inconsistency：There is a discrepancy between the key entities depicted in the textual and visual modalities.\n
# - Event Inconsistency：There is misrepresentation that arises from excessive inference in the written text for the attached video.\n
# - Time & Space Inconsistency：Using unaltered videos that depict past events, but falsely presenting them as recent events.\n
# - Ineffective Visual Information: The vid eo consists of textual information  cannot provide evidence or proof for news content.\n
# - Real News: No form of multimodal misinformation is detected.\n

# Please follow these steps:\n
# 1.Reason step by step through an internal monologue, enclosed within <think> </think> tags.\n
# 2.After reasoning,you can also call the tool when need in the form:<tool> manipulation_detection </tool> to detect manipulation in the video,or call the tool in the form :<tool> diffusion_detection </tool> to detect diffusion in the video.\n
# 3.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> Real News </answer>.\n
# """

# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# for item in data:
#     item_id = item['Id']
#     media_path = find_media_files(base_path, item_id)
#     media_type = get_media_type(media_path)
    
#     # 根据媒体类型选择对应的提示模板
#     if media_type == "image":
#         prompt = image_prompt_template.format(text=item["content"])
#         converted_item = {
#         "images": [media_path],
#         "problem": prompt,
#         "answer": item["label"]
#         }
#     elif media_type == "video":
#         prompt = video_prompt_template.format(text=item["content"])
#         converted_item = {
#         "videos": [media_path],
#         "problem": prompt,
#         "answer": item["label"]
#         }
#     else:
#         # 如果无法确定媒体类型，默认使用图片模板
#         prompt = image_prompt_template.format(text=item["content"])
#         converted_item = {
#             "images": [media_path],
#             "problem": prompt,
#             "answer": item["label"]
#         }
#     converted_data.append(converted_item)

# # 保存转换后的数据
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(converted_data, f, indent=4, ensure_ascii=False)

# print(f"转换完成！共处理 {len(converted_data)} 条数据")



# import json

# def process_accuracy_json(input_file):
#     # 读取JSON文件
#     with open(input_file, 'r') as f:
#         data = json.load(f)
    
#     # 提取所有的input_text和output_text
#     all_input_texts = []
#     all_output_texts = []
    
#     for item in data:
#         all_input_texts.extend(item['input_text'])
#         all_output_texts.extend(item['output_text'])
    
#     # 每5个一组进行分组
#     grouped_data = []
    
#     for i in range(0, len(all_input_texts),5):
#         # 确保有足够的数据
#         if i + 10 <= len(all_input_texts):
#             input_group = all_input_texts[i:i+5]
#             output_group = all_output_texts[i:i+5]
            
#             grouped_data.append({
#                 "input_text": input_group,
#                 "output_text": output_group
#             })
    
#     return grouped_data

# def save_processed_data(output_file, data):
#     # 保存处理后的数据
#     with open(output_file, 'w') as f:
#         json.dump(data, f, indent=2)

# # 主程序
# if __name__ == "__main__":
#     input_file = "/home/chenhui/EasyR1/accuracy_AMG2.json"
#     output_file = "/home/chenhui/EasyR1/processed_accuracy_AMG.json"
    
#     try:
#         # 处理数据
#         processed_data = process_accuracy_json(input_file)
        
#         # 保存结果
#         save_processed_data(output_file, processed_data)
        
#         print(f"处理完成！共生成 {len(processed_data)} 组数据")
#         print(f"结果已保存到: {output_file}")
        
#         # 打印第一组数据作为示例
#         if processed_data:
#             print("\n第一组数据示例:")
#             print(json.dumps(processed_data[0], indent=2))
            
#     except FileNotFoundError:
#         print(f"错误：找不到文件 {input_file}")
#     except json.JSONDecodeError:
#         print(f"错误：{input_file} 不是有效的JSON文件")
#     except Exception as e:
#         print(f"处理过程中发生错误: {str(e)}")


# import json
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 加载accuracy.json文件
# accuracy_file_path = '/home/chenhui/EasyR1/processed_accuracy_AMG.json'
# with open(accuracy_file_path, 'r') as file:
#     accuracy_data = json.load(file)

# # 加载val_1000.json文件
# val_file_path = '/home/chenhui/EasyR1/AMG_test_tool1.json'
# with open(val_file_path, 'r') as file:
#     val_data = json.load(file)

# # 提取accuracy.json中的预测结果
# predicted_answers = []
# print("第19组")
# for item in accuracy_data:
#     output_text = item['output_text'][1]  # 第三个output_text
    
#     # 尝试提取<answer>标签内的内容
#     start = output_text.find('<answer>')
#     end = output_text.find('</answer>')
    
#     if start != -1 and end != -1 and end > start:
#         # 成功找到标签，提取内容
#         start += len('<answer>')
#         predicted_answer = output_text[start:end].strip()
#     else:
#         # 没有找到<answer>标签，默认为' '
#         predicted_answer = ' '
    
#     predicted_answers.append(predicted_answer)

# # 将预测结果保存到result.txt文件
# with open('result.txt', 'w') as file:
#     for answer in predicted_answers:
#         file.write(answer + '\n')

# # 提取val_1000.json中的真实答案
# true_answers = [item['answer'] for item in val_data]

# # 确保两个列表长度一致
# min_length = min(len(true_answers), len(predicted_answers))
# true_answers = true_answers[:min_length]
# predicted_answers = predicted_answers[:min_length]

# print(f"真实答案数量: {len(true_answers)}")
# print(f"预测答案数量: {len(predicted_answers)}")

# # 确保所有预测答案都在预期的标签范围内
# valid_labels = ['Real News', 'Image Fabrication', 'Entity Inconsistency', 'Event Inconsistency','Time & Space Inconsistency','Ineffective Visual Information']

# # 检查是否有预测答案不在有效标签中，如果有则设置为' '
# predicted_answers_cleaned = []
# for answer in predicted_answers:
#     if answer not in valid_labels:
#         predicted_answers_cleaned.append(' ')
#     else:
#         predicted_answers_cleaned.append(answer)

# # 手动计算评估指标
# def calculate_metrics(true_labels, pred_labels, labels):
#     """
#     手动计算Accuracy, Precision, Recall, F1-score
#     """
#     # 初始化每个类别的统计量
#     tp = {label: 0 for label in labels}  # True Positive
#     fp = {label: 0 for label in labels}  # False Positive
#     fn = {label: 0 for label in labels}  # False Negative
    
#     # 计算每个样本的统计量
#     for true, pred in zip(true_labels, pred_labels):
#         for label in labels:
#             if true == label and pred == label:
#                 tp[label] += 1
#             elif true != label and pred == label:
#                 fp[label] += 1
#             elif true == label and pred != label:
#                 fn[label] += 1
    
#     # 计算每个类别的precision, recall, f1
#     precisions = {}
#     recalls = {}
#     f1_scores = {}
    
#     for label in labels:
#         # Precision = TP / (TP + FP)
#         if tp[label] + fp[label] > 0:
#             precisions[label] = tp[label] / (tp[label] + fp[label])
#         else:
#             precisions[label] = 0.0
        
#         # Recall = TP / (TP + FN)
#         if tp[label] + fn[label] > 0:
#             recalls[label] = tp[label] / (tp[label] + fn[label])
#         else:
#             recalls[label] = 0.0
        
#         # F1 = 2 * (Precision * Recall) / (Precision + Recall)
#         if precisions[label] + recalls[label] > 0:
#             f1_scores[label] = 2 * (precisions[label] * recalls[label]) / (precisions[label] + recalls[label])
#         else:
#             f1_scores[label] = 0.0
    
#     # 计算macro平均值
#     macro_precision = sum(precisions.values()) / len(labels)
#     macro_recall = sum(recalls.values()) / len(labels)
#     macro_f1 = sum(f1_scores.values()) / len(labels)
    
#     # 计算accuracy
#     accuracy = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred) / len(true_labels)
    
#     return accuracy, macro_precision, macro_recall, macro_f1, precisions, recalls, f1_scores

# # 计算指标
# accuracy, precision, recall, f1, precisions, recalls, f1_scores = calculate_metrics(
#     true_answers, predicted_answers_cleaned, valid_labels
# )

# # 输出结果
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision (Macro): {precision:.4f}")
# print(f"Recall (Macro): {recall:.4f}")
# print(f"F1 Score (Macro): {f1:.4f}")

# # 生成混淆矩阵
# def print_confusion_matrix(true_labels, pred_labels, labels):
#     """
#     生成并打印混淆矩阵
#     """
#     # 计算混淆矩阵
#     cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
#     # 打印混淆矩阵
#     print("\n混淆矩阵:")
#     print("ground_truth\\prediction", end="")
#     for label in labels:
#         print(f"\t{label}", end="")
#     print()
    
#     for i, true_label in enumerate(labels):
#         print(f"{true_label}({sum(1 for tl in true_labels if tl == true_label)})", end="")
#         for j in range(len(labels)):
#             print(f"\t{cm[i, j]}", end="")
#         print()

# # 打印混淆矩阵
# print_confusion_matrix(true_answers, predicted_answers_cleaned, valid_labels)

# # 可选：可视化混淆矩阵
# def plot_confusion_matrix(true_labels, pred_labels, labels):
#     """
#     绘制并保存混淆矩阵的热力图
#     """
#     cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=labels, yticklabels=labels)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
#     plt.close()
#     print("\n混淆矩阵已保存为 'confusion_matrix.png'")

# # 绘制混淆矩阵热力图
# plot_confusion_matrix(true_answers, predicted_answers_cleaned, valid_labels)

# import json
# import numpy as np

# # 加载文件
# accuracy_file_path = '/home/chenhui/EasyR1/processed_accuracy_AMG.json'
# with open(accuracy_file_path, 'r') as file:
#     accuracy_data = json.load(file)

# val_file_path = '/home/chenhui/EasyR1/AMG_test_tool1.json'
# with open(val_file_path, 'r') as file:
#     val_data = json.load(file)

# # 真实答案
# true_answers = [item['answer'] for item in val_data]
# valid_labels = ['Real News', 'Image Fabrication', 'Entity Inconsistency', 'Event Inconsistency','Time & Space Inconsistency','Ineffective Visual Information']

# # 计算指标函数
# def calculate_metrics(true_labels, pred_labels):
#     accuracy = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred) / len(true_labels)
    
#     # Macro Precision, Recall, F1
#     precisions, recalls, f1s = [], [], []
#     for label in valid_labels:
#         tp = sum(1 for true, pred in zip(true_labels, pred_labels) if true == label and pred == label)
#         fp = sum(1 for true, pred in zip(true_labels, pred_labels) if true != label and pred == label)
#         fn = sum(1 for true, pred in zip(true_labels, pred_labels) if true == label and pred != label)
        
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         precisions.append(precision)
#         recalls.append(recall)
#         f1s.append(f1)
    
#     macro_precision = np.mean(precisions)
#     macro_recall = np.mean(recalls)
#     macro_f1 = np.mean(f1s)
    
#     return accuracy, macro_precision, macro_recall, macro_f1

# # 存储结果
# all_results = []

# # 计算8个output_text的指标
# for text_index in range(5):
#     predicted_answers = []
#     for item in accuracy_data:
#         if len(item['output_text']) > text_index:
#             output_text = item['output_text'][text_index]
#         else:
#             output_text = ""
        
#         start = output_text.find('<answer>')
#         end = output_text.find('</answer>')
        
#         if start != -1 and end != -1 and end > start:
#             start += len('<answer>')
#             predicted_answer = output_text[start:end].strip()
#         else:
#             predicted_answer = ' '
        
#         predicted_answers.append(predicted_answer)
    
#     # 清理预测答案
#     predicted_answers_cleaned = [answer if answer in valid_labels else ' ' for answer in predicted_answers]
    
#     # 计算指标
#     accuracy, precision, recall, f1 = calculate_metrics(true_answers, predicted_answers_cleaned)
#     all_results.append([accuracy, precision, recall, f1])
    
#     print(f"output_text[{text_index}]: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# # 转换为numpy数组便于计算
# results_array = np.array(all_results)

# # 计算均值和方差
# means = np.mean(results_array, axis=0)
# stds = np.std(results_array, axis=0)

# print("\n8组指标的平均值:")
# print(f"Accuracy: {means[0]:.4f}, Precision: {means[1]:.4f}, Recall: {means[2]:.4f}, F1: {means[3]:.4f}")

# print("\n8组指标的方差:")
# print(f"Accuracy: {stds[0]:.4f}, Precision: {stds[1]:.4f}, Recall: {stds[2]:.4f}, F1: {stds[3]:.4f}")



# import os
# from PIL import Image
# from pathlib import Path

# def delete_corrupted_images(directory_path):
#     """删除指定目录下所有损坏的图片文件"""
#     image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             file_path = Path(root) / file
#             if file_path.suffix.lower() in image_extensions:
#                 try:
#                     with Image.open(file_path) as img:
#                         img.verify()
#                     with Image.open(file_path) as img:
#                         img.load()
#                 except Exception as e:
#                     print(f"删除损坏文件: {file_path}")
#                     os.remove(file_path)

# if __name__ == "__main__":
#     target_dir = "/home/chenhui/EasyR1/AMG/__MACOSX/AMG_MEDIA/test"
#     delete_corrupted_images(target_dir)
#     print("损坏图片清理完成")

# import os
# from PIL import Image
# from pathlib import Path

# def detect_corrupted_images(directory_path):
#     """检测指定目录下所有损坏的图片文件并统计图片数量"""
#     image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
#     corrupted_files = []
#     total_images = 0
#     valid_images = 0
    
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             file_path = Path(root) / file
#             if file_path.suffix.lower() in image_extensions:
#                 total_images += 1
#                 try:
#                     with Image.open(file_path) as img:
#                         img.verify()
#                     with Image.open(file_path) as img:
#                         img.load()
#                     valid_images += 1
#                 except Exception as e:
#                     print(f"损坏文件: {file_path}")
#                     corrupted_files.append(str(file_path))
    
#     print(f"\n检测完成！")
#     print(f"总图片数量: {total_images}")
#     print(f"有效图片数量: {valid_images}")
#     print(f"损坏图片数量: {len(corrupted_files)}")
    
#     return corrupted_files, total_images, valid_images

# if __name__ == "__main__":
#     target_dir = "/home/chenhui/EasyR1/AMG/AMG_MEDIA/train"
#     corrupted_files, total, valid = detect_corrupted_images(target_dir)
# import json

# def delete_mp4_entries(json_file_path):
#     """删除JSON文件中images字段为mp4结尾的数据"""
    
#     # 读取JSON文件
#     with open(json_file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 过滤掉images字段中包含.mp4文件的条目
#     filtered_data = []
#     deleted_count = 0
    
#     for item in data:
#         images = item.get('images', [])
#         has_mp4 = any(image.lower().endswith('.mp4') for image in images)
        
#         if has_mp4:
#             deleted_count += 1
#             print(f"删除包含MP4的条目: {images}")
#         else:
#             filtered_data.append(item)
    
#     # 保存处理后的数据
#     with open(json_file_path, 'w', encoding='utf-8') as f:
#         json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
#     print(f"\n处理完成！")
#     print(f"原始条目数: {len(data)}")
#     print(f"删除条目数: {deleted_count}")
#     print(f"剩余条目数: {len(filtered_data)}")

# if __name__ == "__main__":
#     json_file_path = "/home/chenhui/EasyR1/AMG_test_tool1.json"
#     delete_mp4_entries(json_file_path)
# import json
# import random

# # 读取原始文件
# with open('/home/chenhui/EasyR1/AMG_train_tool1.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 定义需要提取的misinformation类型
# target_misinfo_types = [
#     "Event Inconsistency",
#     "Image Fabrication", 
#     "Time & Space Inconsistency",
#     "Ineffective Visual Information",
#     "Entity Inconsistency"
# ]

# # 提取所有目标misinformation类型的数据
# misinfo_data = []
# real_news_data = []

# for item in data:
#     answer = item.get('answer', '')
#     if answer in target_misinfo_types:
#         misinfo_data.append(item)
#     elif answer == "Real News":
#         real_news_data.append(item)

# print(f"找到的misinformation数据条数: {len(misinfo_data)}")
# print(f"找到的Real News数据条数: {len(real_news_data)}")

# # 随机抽取552条Real News数据
# if len(real_news_data) >= 552:
#     selected_real_news = random.sample(real_news_data, 552)
# else:
#     print(f"警告: Real News数据不足552条，只有{len(real_news_data)}条，将使用所有可用的Real News数据")
#     selected_real_news = real_news_data

# # 合并数据
# combined_data = misinfo_data + selected_real_news

# print(f"合并后的总数据条数: {len(combined_data)}")
# print(f"- Misinformation数据: {len(misinfo_data)}")
# print(f"- Real News数据: {len(selected_real_news)}")

# # 统计各类别的数量
# category_count = {}
# for item in combined_data:
#     answer = item.get('answer', '')
#     category_count[answer] = category_count.get(answer, 0) + 1

# print("各类别数据分布:")
# for category, count in category_count.items():
#     print(f"  {category}: {count}条")

# # 保存为新文件
# output_file = '/home/chenhui/EasyR1/AMG_train_tool3.json'
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(combined_data, f, ensure_ascii=False, indent=2)

# print(f"\n数据已保存到: {output_file}")

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
#     input_path = "/home/chenhui/EasyR1/AMG-An-Attributing-Multi-modal-Fake-News-Dataset/dataset/train.json"
#     output_path = "/home/chenhui/EasyR1/AMG_data/AMG_2000.json"
    
#     success = extract_random_entries_advanced(input_path, output_path, 2000)
    
#     if success:
#         print("\n操作完成！")
#     else:
#         print("\n操作失败！")