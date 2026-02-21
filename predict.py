# coding=utf-8
import os
import torch.utils.data
from data_utils import TrainDatasetFromFolder, TestDatasetFromFolder, calMetric_iou
from models.CTDMNet.network import CDNet
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import yaml
import swanlab

parser = argparse.ArgumentParser(description='Test Change Detection Models')
parser.add_argument('--config', default='config_fusion/JL1.yaml', type=str, help='path to config')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--model_dir', default='Trained_model/JL1/best_model.pth', required=True, type=str, help='Path to model checkpoint')
parser.add_argument('--n_class', default=9, type=int, help='the number of types of change')
parser.add_argument('--save_dir', default='results/JL1_2023/simple_test', required=True, type=str, help='directory to save results')
parser.add_argument('--img_size', type=int, default=256, help='input image size')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print('Result directory created!')

# 初始化模型
with open(args.config) as f:
    opt = yaml.safe_load(f)
swanlab.init(mode='disabled', project='1', config=opt)
config = swanlab.config
# 修改CDNet初始化
netCD = CDNet(n_class=args.n_class, config=config).to(device)
netCD.load_state_dict(torch.load(args.model_dir, weights_only=False)['model_state_dict'])
netCD.eval()

if __name__ == '__main__':
    test_set = TestDatasetFromFolder(config, config['path_img1'], config['path_img2'],
                                    config['path_lab'], config['test_txt'], norm_type=config['norm_type'])
    test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=1, shuffle=False)
    
    for hr_img1, lr_img2, label, image_name in tqdm(test_loader):
        orig_h, orig_w = hr_img1.shape[-2:]
        # 使用配置中的img_size作为目标尺寸
        new_h = config['img_size']
        new_w = config['img_size']
        
        # 调整图像大小
        hr_img1 = F.interpolate(hr_img1.to(device, dtype=torch.float), (new_h, new_w), mode='bilinear', align_corners=True)
        lr_img2 = F.interpolate(lr_img2.to(device, dtype=torch.float), (new_h, new_w), mode='bilinear', align_corners=True)
        
        with torch.no_grad():
            # 预测
            result = netCD(hr_img1, lr_img2)
            # 将预测结果恢复到原始尺寸
            if config.get('multiplier') is not None:
                result = F.interpolate(result, (orig_h, orig_w), mode='bilinear', align_corners=True)
            pred = torch.argmax(result, 1).cpu().numpy()[0]  # [H, W]
            
            # 保存原始预测结果
            pred_pil = Image.fromarray(pred.astype('uint8'))
            pred_pil.save(os.path.join(args.save_dir, f"{image_name[0]}"))