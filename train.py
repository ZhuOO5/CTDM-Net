# coding=utf-8
# 这里是部分冻结参数的情况，把冻结的层数作为一个超参
from math import exp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numpy import save
import yaml
import swanlab
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder, DA_DatasetFromFolder, WODA_DatasetFromFolder
from models.CTDMNet.network import CDNet
from loss.losses import cross_entropy
from metrics import update_seg_metrics, get_seg_metrics, eval_metrics
import numpy as np
import random


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_exp_name(cfg, config_path):
    from datetime import datetime
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    name = f"Fusion_dif" \
           f"_{time_str}"
    return name

def train(config_path):
    # 加载配置文件
    with open(config_path) as f:
        opt = yaml.safe_load(f)
    exp_name = make_exp_name(opt,config_path)
    opt['exp_dir'] = os.path.join(opt['save_dir'], exp_name)
    
    swanlab.init(mode='cloud',project="train2021_JL1", config=opt)
    # wandb.init(mode='cloud',config=opt)
    config = swanlab.config
    
    # 设备设置isabled
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['exp_dir'], exist_ok=True)

    # 数据加载（保持原逻辑）
    if config['Train_with_aug']:
        train_set = DA_DatasetFromFolder(config, config['path_img1'], config['path_img2'], 
                                       config['path_lab'], config['train_txt'], crop=False, 
                                       norm_type=config['norm_type'],img_size=config['img_size'])
    else:
        train_set = WODA_DatasetFromFolder(config, config['path_img1'], config['path_img2'],
                                         config['path_lab'], config['train_txt'],
                                         crop=False,
                                         norm_type=config['norm_type'],img_size=config['img_size'])
    
    train_loader = DataLoader(train_set, num_workers=24, batch_size=config['batchsize'], shuffle=True)
    val_loader = DataLoader(TrainDatasetFromFolder(config, config['path_img1'], config['path_img2'],
                                                 config['path_lab'], config['val_txt']),
                          num_workers=24, batch_size=config['batchsize'])
    test_set = TrainDatasetFromFolder(config, config['path_img1'], config['path_img2'],
    config['path_lab'], config['test_txt'])
    test_loader = DataLoader(test_set, num_workers=24, batch_size=config['batchsize'])
    # 模型初始化
    netCD = CDNet(config, n_class=config['n_class'],).to(device)
    # if torch.cuda.device_count() > 1:
    #     netCD = torch.nn.DataParallel(netCD)
    
    if config['pretrained']:
        netCD.load_state_dict(torch.load(config['pretrain_path']))

    backbone_state_dict = torch.load(config['DINOv2_pretrain_path'])
    netCD.backbone1.load_state_dict(backbone_state_dict)

    if(config['froze_DINOv2'] == True):
        for name, param in netCD.backbone1.named_parameters():
            param.requires_grad = False  # 默认冻结DINOv2所有参数

        
    # 优化器和损失函数
    optimizer = torch.optim.Adam(netCD.parameters(), lr=0.0001, betas=(0.9, 0.999))
    criterion = cross_entropy(ignore_index=255).to(device)

    # 训练循环
    best_miou = 0
    for epoch in range(1, config['num_epochs'] + 1):
        netCD.train()
        train_loss = 0
        
        # 训练阶段
        for img1, img2, lbl in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            img1, img2, lbl = img1.to(device), img2.to(device), lbl.to(device)
            
            optimizer.zero_grad()
            prob = netCD(img1, img2, lbl)
            # loss = criterion(prob, torch.argmax(lbl, 1))
            loss = criterion(prob, lbl)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            swanlab.log({"batch_loss": loss.item()})
        
        # 验证阶段
        # m_items_test = m_items.clone()
        val_metrics = validate(netCD, val_loader, device, config)
        
        # 记录指标
        avg_train_loss = train_loss / len(train_loader)
        log_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_mIoU": val_metrics['mIoU'],
            "val_mF1": val_metrics['mF1'],
            "val_Fscd": val_metrics['Fscd'],
            "val_pixAcc": val_metrics['pixAcc'],
            "val_class_IoU": val_metrics['class_IoU'],
            "val_class_F1": val_metrics['class_F1']
        }
        swanlab.log(log_data)


        # 保存最佳模型
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            # 保存最佳模型和epoch信息
            torch.save({
                'epoch': epoch,
                'model_state_dict': netCD.state_dict(),
                'best_miou': best_miou
            }, f"{config['exp_dir']}/best_model.pth")
            # 记录最佳epoch
            with open(f"{config['exp_dir']}/best_epoch.txt", 'w') as f:
                f.write(f"Best Epoch: {epoch}, mIoU: {best_miou}")

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': netCD.state_dict(),
        'best_miou': best_miou
    }, f"{config['exp_dir']}/final_model.pth")
    # 存储最终模型评估效果
    test_metrics = validate(netCD, test_loader, device, config)
    test_log_data = {
        # "best_epoch": epoch,
        "last_test_mIoU": test_metrics['mIoU'],
        "last_test_mF1": test_metrics['mF1'],
        "last_test_Fscd": test_metrics['Fscd'],
        "last_test_pixAcc": test_metrics['pixAcc'],
        "last_test_class_IoU": test_metrics['class_IoU'],
        "last_test_class_F1": test_metrics['class_F1'] 
    }
    swanlab.log(test_log_data)

    swanlab.finish()

def validate(model, val_loader, device, config):
    model.eval()
    # total_inter, total_union = 0, 0
    total_inter = np.zeros(config['n_class'], dtype=np.float64)
    total_union = np.zeros(config['n_class'], dtype=np.float64)
    total_correct, total_label = 0, 0
    total_FPR, total_pred = 0, 0

    with torch.no_grad():
        for img1, img2, lbl in tqdm(val_loader, desc='Validating'):
            # 图像大小跟训练保持一致
            orig_h, orig_w = img1.shape[-2:]
            multiplier = config['multiplier']
            # new_h = int(orig_h / multiplier + 0.5) * multiplier
            # new_w = int(orig_w / multiplier + 0.5) * multiplier
            new_h = config['img_size']
            new_w = config['img_size']
            
            img1 = F.interpolate(img1.to(device), (new_h, new_w), mode='bilinear', align_corners=True)
            img2 = F.interpolate(img2.to(device), (new_h, new_w), mode='bilinear', align_corners=True)
            lbl = lbl.to(device)

            prob = model(img1, img2)
            if multiplier is not None:
                prob = F.interpolate(prob, (orig_h, orig_w), mode='bilinear', align_corners=True)

            # seg_metrics = eval_metrics(prob, torch.argmax(lbl, 1), config['n_class'])
            seg_metrics = eval_metrics(prob, lbl, config['n_class'])
            total_inter, total_union, total_correct, total_label, total_FPR, total_pred = update_seg_metrics(total_inter, total_union, total_correct, 
                             total_label, total_FPR, total_pred, *seg_metrics)

    metrics = get_seg_metrics(total_correct, total_label, total_inter, 
                             total_union, total_FPR, total_pred, config['n_class'])
    return {
        'pixAcc': metrics['Pixel_Accuracy'],
        'mIoU': metrics['Mean_IoU'],
        'Fscd': metrics['Fscd'],
        'mF1': metrics['Mean_F1'],
        'class_F1':metrics['Class_F1'],
        'class_IoU':metrics['Class_IoU']
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    
    train(args.config)
