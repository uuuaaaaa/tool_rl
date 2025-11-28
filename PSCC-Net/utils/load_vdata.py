# import os

# import numpy as np
# import torch.utils.data as data
# import torch
# import imageio
# import json
# # 在 load_vdata.py 文件开头添加
# from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import logging



# class TestData(data.Dataset):
#     # def __init__(self, args):
#     #     super(TestData, self).__init__()

#     #     ddir = '/home/chenhui/EasyR1/AMG_train_tool1.json'
#     #     names = os.listdir(ddir)
#     #     authentic_names = [os.path.join(ddir, name) for name in names if 'authentic' in name]
#     #     authentic_class = [0] * len(authentic_names)

#     #     fake_names = [os.path.join(ddir, name) for name in names if 'authentic' not in name]
#     #     fake_class = [1] * len(fake_names)

#     #     self.image_names = authentic_names + fake_names
#     #     self.image_class = authentic_class + fake_class
#     def __init__(self, args):
#         super(TestData, self).__init__()

#         # 读取 JSON 文件
#         json_path = '/home/chenhui/EasyR1/AMG_train_tool1.json'
#         with open(json_path, 'r') as f:
#             data = json.load(f)
        
#         # 提取所有图像路径
#         self.image_names = []
#         self.image_class = []
        
#         for item in data:
#             # 获取图像路径列表
#             image_paths = item["images"]
#             # 这里假设每个item只有一个图像，如果有多个则取第一个
#             if image_paths:
#                 image_path = image_paths[0]
#                 self.image_names.append(image_path)
                
#                 # 根据answer判断类别
#                 answer = item.get("answer", "").strip()
#                 if answer == "Real News":
#                     self.image_class.append(0)  # authentic
#                 else:
#                     self.image_class.append(1)  # fake
        
#         print(f"总共加载了 {len(self.image_names)} 张图像")
#         print(f"真实新闻数量: {self.image_class.count(0)}")
#         print(f"虚假新闻数量: {self.image_class.count(1)}")
#     def rgba2rgb(self, rgba, background=(255, 255, 255)):
#         row, col, ch = rgba.shape

#         rgb = np.zeros((row, col, 3), dtype='float32')
#         r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

#         a = np.asarray(a, dtype='float32') / 255.0

#         R, G, B = background

#         rgb[:, :, 0] = r * a + (1.0 - a) * R
#         rgb[:, :, 1] = g * a + (1.0 - a) * G
#         rgb[:, :, 2] = b * a + (1.0 - a) * B

#         return np.asarray(rgb, dtype='uint8')

#     # def get_item(self, index):
#     #     image_name = self.image_names[index]
#     #     cls = self.image_class[index]

#     #     image = imageio.imread(image_name)

#     #     if image.shape[-1] == 4:
#     #         image = self.rgba2rgb(image)

#     #     image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

#     #     return image, cls, image_name
#     # 修改 get_item 方法，添加错误处理

#     def get_item(self, index):
#         image_name = self.image_names[index]
#         cls = self.image_class[index]

#         try:
#             # 尝试读取图像
#             image = imageio.imread(image_name)
            
#             if image.shape[-1] == 4:
#                 image = self.rgba2rgb(image)
                
#             image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)
            
#             return image, cls, image_name
            
#         except OSError as e:
#             print(f"警告: 图像文件损坏，跳过文件: {image_name}, 错误: {str(e)}")
#             # 返回一个占位符图像或跳过
#             # 这里我们创建一个黑色图像作为占位符
#             placeholder_image = np.zeros((256, 256, 3), dtype=np.uint8)
#             image = torch.from_numpy(placeholder_image.astype(np.float32) / 255).permute(2, 0, 1)
#             return image, cls, image_name + " [CORRUPTED]"
            
#         except Exception as e:
#             print(f"警告: 处理图像时出错，跳过文件: {image_name}, 错误: {str(e)}")
#             # 返回一个占位符图像
#             placeholder_image = np.zeros((256, 256, 3), dtype=np.uint8)
#             image = torch.from_numpy(placeholder_image.astype(np.float32) / 255).permute(2, 0, 1)
#             return image, cls, image_name + " [ERROR]"

#     def __getitem__(self, index):
#         res = self.get_item(index)
#         return res

#     def __len__(self):
#         return len(self.image_names)

import os

import numpy as np
import torch.utils.data as data
import torch
import imageio


class TestData(data.Dataset):
    def __init__(self, args):
        super(TestData, self).__init__()

        ddir = './sample'
        names = os.listdir(ddir)
        authentic_names = [os.path.join(ddir, name) for name in names if 'authentic' in name]
        authentic_class = [0] * len(authentic_names)

        fake_names = [os.path.join(ddir, name) for name in names if 'authentic' not in name]
        fake_class = [1] * len(fake_names)

        self.image_names = authentic_names + fake_names
        self.image_class = authentic_class + fake_class

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def get_item(self, index):
        image_name = self.image_names[index]
        cls = self.image_class[index]

        image = imageio.imread(image_name)

        if image.shape[-1] == 4:
            image = self.rgba2rgb(image)

        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        return image, cls, image_name

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return len(self.image_names)