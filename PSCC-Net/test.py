# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# from utils.utils import save_image

# from models.seg_hrnet import get_seg_model
# from models.seg_hrnet_config import get_hrnet_cfg
# from utils.config import get_pscc_args
# from models.NLCDetection import NLCDetection
# from models.detection_head import DetectionHead
# from utils.load_vdata import TestData

# device_ids = [Id for Id in range(torch.cuda.device_count())]
# device = torch.device('cuda:0')
# import json

# def load_network_weight(net, checkpoint_dir, name):
#     weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
#     net_state_dict = torch.load(weight_path, map_location='cuda:0')
#     net.load_state_dict(net_state_dict)
#     print('{} weight-loading succeeds'.format(name))


# def test(args):
#     # define backbone
#     FENet_name = 'HRNet'
#     FENet_cfg = get_hrnet_cfg()
#     FENet = get_seg_model(FENet_cfg)

#     # define localization head
#     SegNet_name = 'NLCDetection'
#     SegNet = NLCDetection(args)

#     # define detection head
#     ClsNet_name = 'DetectionHead'
#     ClsNet = DetectionHead(args)

#     FENet_checkpoint_dir = './PSCC-Net/checkpoint/{}_checkpoint'.format(FENet_name)
#     SegNet_checkpoint_dir = './PSCC-Net/checkpoint/{}_checkpoint'.format(SegNet_name)
#     ClsNet_checkpoint_dir = './PSCC-Net/checkpoint/{}_checkpoint'.format(ClsNet_name)

#     # load FENet weight
#     FENet = FENet.to(device)
#     FENet = nn.DataParallel(FENet, device_ids=device_ids)
#     load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

#     # load SegNet weight
#     SegNet = SegNet.to(device)
#     SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
#     load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

#     # load ClsNet weight
#     ClsNet = ClsNet.to(device)
#     ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
#     load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

#     test_data_loader = DataLoader(TestData(args), batch_size=1, shuffle=False,
#                                   num_workers=8)
#     results = []
#     error_files = []  
#     for batch_id, test_data in enumerate(test_data_loader):
#         try:
#             image, cls, name = test_data
#             image = image.to(device)

#             with torch.no_grad():

#                 # backbone network
#                 FENet.eval()
#                 feat = FENet(image)

#                 # localization head
#                 SegNet.eval()
#                 pred_mask = SegNet(feat)[0]

#                 pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
#                                         align_corners=True)

#                 # classification head
#                 ClsNet.eval()
#                 pred_logit = ClsNet(feat)

#             # ce
#             sm = nn.Softmax(dim=1)
#             pred_logit = sm(pred_logit)
#             _, binary_cls = torch.max(pred_logit, 1)

#             pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

#             if args.save_tag:
#                 save_image(pred_mask, name, 'mask')

#             print_name = name[0]
#             image_path = name[0]  # 获取完整的图像路径.split('/')[-1].split('.')[0]
#             print(f'The image {print_name} is {pred_tag}')
            
#             # 构建结果字典
#             result_item = {
#                 image_path: {
#                     "prediction": pred_tag
#                 }
#             }
#             results.append(result_item)
                
#         except OSError as e:
#             # 捕获图像文件错误
#             error_msg = str(e)
#             image_path = name[0] if 'name' in locals() else f'unknown_{batch_id}'
#             print(f"错误: 跳过文件 {image_path}, 错误信息: {error_msg}")
            
#             # 记录错误文件信息
#             error_info = {
#                 "file_path": image_path,
#                 "batch_id": batch_id,
#                 "error": error_msg
#             }
#             error_files.append(error_info)
#             continue
            
#         except Exception as e:
#             # 捕获其他可能的错误
#             error_msg = str(e)
#             image_path = name[0] if 'name' in locals() else f'unknown_{batch_id}'
#             print(f"未知错误: 跳过文件 {image_path}, 错误信息: {error_msg}")
            
#             error_info = {
#                 "file_path": image_path,
#                 "batch_id": batch_id,
#                 "error": error_msg
#             }
#             error_files.append(error_info)
#             continue

#     # 保存检测结果到JSON文件
#     output_file = 'detection_results_AMG.json'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
    
#     # 保存错误文件信息
#     if error_files:
#         error_output_file = 'error_files.json'
#         with open(error_output_file, 'w', encoding='utf-8') as f:
#             json.dump(error_files, f, indent=4, ensure_ascii=False)
#         print(f'\n错误文件信息已保存到: {error_output_file}')
#         print(f'总共遇到 {len(error_files)} 个错误文件')

#     print(f'\n测试结果已保存到: {output_file}')
#     print(f'成功处理了 {len(results)} 张图像')
#     print(f'跳过了 {len(error_files)} 张损坏或错误的图像')

#     return results, error_files


# if __name__ == '__main__':
#     args = get_pscc_args()
#     test(args)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import save_image

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.load_vdata import TestData

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda:0')


def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    net_state_dict = torch.load(weight_path, map_location='cuda:0')
    net.load_state_dict(net_state_dict)
    print('{} weight-loading succeeds'.format(name))


def test(args):
    # define backbone
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    # define localization head
    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    # define detection head
    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    SegNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
    ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)

    # load FENet weight
    FENet = FENet.to(device)
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)

    # load SegNet weight
    SegNet = SegNet.to(device)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    load_network_weight(SegNet, SegNet_checkpoint_dir, SegNet_name)

    # load ClsNet weight
    ClsNet = ClsNet.to(device)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)

    test_data_loader = DataLoader(TestData(args), batch_size=1, shuffle=False,
                                  num_workers=8)

    for batch_id, test_data in enumerate(test_data_loader):

        image, cls, name = test_data
        image = image.to(device)

        with torch.no_grad():

            # backbone network
            FENet.eval()
            feat = FENet(image)

            # localization head
            SegNet.eval()
            pred_mask = SegNet(feat)[0]

            pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                      align_corners=True)

            # classification head
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        # ce
        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)

        pred_tag = 'forged' if binary_cls.item() == 1 else 'authentic'

        if args.save_tag:
            save_image(pred_mask, name, 'mask')

        print_name = name[0].split('/')[-1].split('.')[0]

        print(f'The image {print_name} is {pred_tag}')


if __name__ == '__main__':
    args = get_pscc_args()
    test(args)