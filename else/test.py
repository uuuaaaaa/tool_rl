# import json

# def process_data(input_file, output_file):
#     # 读取原始JSON文件
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     result = []
    
#     for item in data:
#         # 提取images路径
#         image_path = item['images'][0]  # 取第一个image路径
#         answer = item['answer']
        
#         # 根据answer值确定output内容
#         if answer == 'textual_veracity_distortion':
#             output_text = "there are factual errors in the news caption"
#         else:
#             output_text = "there are not any factual errors in the news caption"
        
#         # 构建结果格式
#         result_item = {
#             image_path: {
#                 "output": output_text
#             }
#         }
#         result.append(result_item)
    
#     # 保存结果到JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)
    
#     print(f"处理完成！共处理 {len(result)} 条数据")
#     print(f"结果已保存到: {output_file}")

# # 使用示例
# input_file = "/home/chenhui/EasyR1/train_10000tool.json"
# output_file = "search_results.json"

# process_data(input_file, output_file)

import json
import os

# 文件路径设置
# input_file = "/home/chenhui/EasyR1/train_1200.json"
# output_file = "/home/chenhui/EasyR1/train_1200grpo.json"
# base_path = "/home/chenhui/EasyR1/MMFakeBench_test"

input_file = "/home/chenhui/EasyR1/MMFakeBench_val/source/MMFakeBench_val.json"
output_file = "/home/chenhui/EasyR1/val_1000grpo.json"
base_path = "/home/chenhui/EasyR1/MMFakeBench_val"

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换数据
converted_data = []
prompt_template = """<image>The news caption is: {text}
Given a news caption and an accompanying image, your task is to classify the type of multimodal misinformation.\n
The four possible categories are:\n
- textual_veracity_distortion: There are factual errors in the news caption.\n
- visual_veracity_distortion: The image is AI-generated, manipulated, or contradicts objective reality.\n
- mismatch: The news caption is extremely inconsistent with the content of the image.\n
- original: No form of multimodal misinformation is detected.\n

Please follow these steps:\n
1.Reason step by step through an internal monologue, enclosed within <think> </think> tags.\n
2.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> original </answer>.\n
 """
# 2.After reasoning, you may use the following tools when need:
# - Use `<search>{text}</search>` to verify factual claims in the news caption.
# - Use `<tool>manipulation_detection</tool>` to check for image manipulation.
# - Use `<tool>diffusion_detection</tool>` to detect if the image is AI-generated.

# 2. If needed, you may use **one** of the following tools during your reasoning:
#    - Use `<search>{text}</search>` to verify factual claims in the news caption.
#    - Use `<tool>manipulation_detection</tool>` to check for image manipulation.
#    - Use `<tool>diffusion_detection</tool>` to detect if the image is AI-generated.
# 3.Based on the above information,provide your answer in the format: <answer> type of multimodal misinformation</answer>,for example: <answer> original </answer>.\n

# 2.After reasoning,you can call the search engine when need in the form: <search>{text}</search> to detect factual errors in the news caption.\n
# 3.After reasoning,you can also call the tool when need in the form:<tool> manipulation_detection </tool> to detect manipulation in the image,or call the tool in the form :<tool> diffusion_detection </tool> to detect diffusion in the image.\n
for item in data:
    converted_item = {
        "images": [os.path.join(base_path, item["image_path"].lstrip('/'))],
        "problem": prompt_template.format(text=item["text"]),
        "answer": item["fake_cls"]
    }
    converted_data.append(converted_item)

# 保存转换后的数据
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)

print(f"转换完成！共处理 {len(converted_data)} 条数据")


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
    
#     for i in range(0, len(all_input_texts),10):
#         # 确保有足够的数据
#         if i + 5 <= len(all_input_texts):
#             input_group = all_input_texts[i:i+10]
#             output_group = all_output_texts[i:i+10]
            
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
#     input_file = "/home/chenhui/EasyR1/MMFakeBench_data/result/accuracy1tool3_0.82.json"
#     output_file = "/home/chenhui/EasyR1/MMFakeBench_tool3_acc.json"
    
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


# import re
# import matplotlib.pyplot as plt
# import numpy as np

# # 文件路径
# file_path = '/home/chenhui/EasyR1/loss.txt'

# # 初始化数据存储
# rewards_overall = []
# rewards_format = []
# rewards_accuracy = []

# # 读取文件并提取数据
# with open(file_path, 'r') as file:
#     content = file.read()

# # 使用正则表达式匹配所有的 reward 数据
# pattern = r"reward:\{[^}]*\}"
# matches = re.findall(pattern, content)

# for match in matches:
#     # 提取 np.float64() 中的数值
#     overall_match = re.search(r"'reward/overall': np\.float64\(([\d\.]+)\)", match)
#     format_match = re.search(r"'reward/format': np\.float64\(([\d\.]+)\)", match)
#     accuracy_match = re.search(r"'reward/accuracy': np\.float64\(([\d\.]+)\)", match)
    
#     if overall_match:
#         rewards_overall.append(float(overall_match.group(1)))
#     if format_match:
#         rewards_format.append(float(format_match.group(1)))
#     if accuracy_match:
#         rewards_accuracy.append(float(accuracy_match.group(1)))

# # 绘制图像
# plt.figure(figsize=(12, 8))

# # 绘制三条曲线
# plt.plot(rewards_overall, label='reward/overall', marker='o', linewidth=2, markersize=4)
# plt.plot(rewards_format, label='reward/format', marker='s', linewidth=2, markersize=4)
# plt.plot(rewards_accuracy, label='reward/accuracy', marker='^', linewidth=2, markersize=4)

# # 添加标题和标签
# plt.title('Reward Metrics Over Steps', fontsize=16, fontweight='bold')
# plt.xlabel('Step', fontsize=14)
# plt.ylabel('Reward Value', fontsize=14)
# plt.legend(fontsize=12)

# # 设置坐标轴范围
# plt.ylim(0, 1.5)  # 根据数据范围调整

# # 显示网格
# plt.grid(True, alpha=0.3)

# # 添加一些美化
# plt.tight_layout()

# # 保存图像到当前目录
# plt.savefig('reward_metrics.png', dpi=300, bbox_inches='tight')

# # 显示图像
# plt.show()

# # 打印一些统计信息
# print(f"总共 {len(rewards_overall)} 个数据点")
# print(f"reward/overall 平均值: {np.mean(rewards_overall):.4f}")
# print(f"reward/format 平均值: {np.mean(rewards_format):.4f}")
# print(f"reward/accuracy 平均值: {np.mean(rewards_accuracy):.4f}")



# import json
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 加载accuracy.json文件
# accuracy_file_path = "/home/chenhui/EasyR1/MMFakeBench_tool3_acc.json"
# with open(accuracy_file_path, 'r') as file:
#     accuracy_data = json.load(file)

# # 加载val_1000.json文件
# val_file_path = '/home/chenhui/EasyR1/MMFakeBench_data/val_1000grpo.json'
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
#         # 没有找到<answer>标签，默认为'original'
#         predicted_answer = ' '
    
#     predicted_answers.append(predicted_answer)

# # 将预测结果保存到result.txt文件
# with open('result.txt', 'w') as file:
#     for answer in predicted_answers:
#         file.write(answer + '\n')

# # 提取val_1000.json中的真实答案
# true_answers = [item['answer'] for item in val_data]

# # 确保所有预测答案都在预期的标签范围内
# valid_labels = ['original', 'mismatch', 'textual_veracity_distortion', 'visual_veracity_distortion']

# # 检查是否有预测答案不在有效标签中，如果有则设置为'original'
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

# def simple_check_answer_tags(file_path):
#     """
#     简化版检查函数，只返回是否存在<answer>标签的布尔值列表
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         results = []
        
#         for item in data:
#             item_results = []
#             for j in range(1):
#                 if j < len(item['output_text']):
#                     text = item['output_text'][j]
#                     has_tag = '<answer>' in text and '</answer>' in text
#                     item_results.append(has_tag)
#                 else:
#                     item_results.append(False)
#             results.append(item_results)
        
#         return results
        
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# # 使用简化版
# file_path = "/home/chenhui/EasyR1/MMFakeBench_tool1_acc.json"
# results = simple_check_answer_tags(file_path)
# a=0
# if results:
#     for i, item_results in enumerate(results):
#         a=a+sum(item_results)
#         print(f"Item {i}: {item_results}")
#         print(f"  包含标签的数量: {sum(item_results)}")
#     print(a)

# import json
# import numpy as np

# # 加载文件
# accuracy_file_path = "/home/chenhui/EasyR1/MMFakeBench_tool1_acc.json"
# with open(accuracy_file_path, 'r') as file:
#     accuracy_data = json.load(file)

# val_file_path = '/home/chenhui/EasyR1/MMFakeBench_data/val_1000grpo.json'
# with open(val_file_path, 'r') as file:
#     val_data = json.load(file)

# # 真实答案
# true_answers = [item['answer'] for item in val_data]
# valid_labels = ['original', 'mismatch', 'textual_veracity_distortion', 'visual_veracity_distortion']

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
