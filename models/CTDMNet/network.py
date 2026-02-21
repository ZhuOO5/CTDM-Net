import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .Memory import *
from loss.losses import cross_entropy

# -----------------------
from models.CTDMNet.backbone.dinov2 import DINOv2
from models.CTDMNet.util.blocks import FeatureFusionBlock, _make_scratch
from models.CTDMNet.backbone.Mobilenetv3 import MobileNetV3Backbone
from models.CTDMNet.dif_module.dif import ChangeInformationModel
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 通道维度的平均池化和最大池化
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度池化，保留空间尺寸 H × W
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        # 在通道维度拼接
        pooled = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 2, H, W]
        # 通过卷积生成空间注意力图
        attention = self.conv(pooled)  # [batch_size, 1, H, W]
        attention = self.sigmoid(attention)
        return x * attention  # 逐元素相乘，输出 [batch_size, in_channels, H, W]

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=None):
        super(SKConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 动态设置 reduction 值
        if reduction is None:
            reduction = 4 if out_channels <= 64 else 8 if out_channels <= 128 else 16
        self.reduction = max(4, reduction)  # 确保 reduction 至少为 4

        # 定义三个共享的 3x3 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 空间注意力模块，分别应用于三个分支
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention(out_channels, kernel_size=7),
            SpatialAttention(out_channels, kernel_size=7),
            SpatialAttention(out_channels, kernel_size=7)
        ])

        # 全局空间注意力模块，作用于求和后的特征
        self.global_spatial_attention = SpatialAttention(out_channels, kernel_size=7)

        # 全局池化层和全连接层，用于选择性核机制
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, max(8, out_channels // self.reduction)),  # 最小维度为 8
            nn.ReLU(inplace=True),
            nn.Linear(max(8, out_channels // self.reduction), out_channels * 3),  # 3 个分支
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 串联计算三个 3x3 卷积
        out1 = self.conv1(x)  # 感受野 3x3
        out2 = self.conv2(out1)  # 感受野 5x5
        out3 = self.conv3(out2)  # 感受野 7x7

        # 分别应用空间注意力
        out1 = self.spatial_attentions[0](out1)
        out2 = self.spatial_attentions[1](out2)
        out3 = self.spatial_attentions[2](out3)

        # 合并分支
        outs = [out1, out2, out3]  # 每个 out: [batch_size, out_channels, H, W]
        
        # 对空间注意力后的分支求和
        U = sum(outs)  # [batch_size, out_channels, H, W]
        
        # 应用全局空间注意力（可选，原始代码中被注释）
        # U = self.global_spatial_attention(U)  # [batch_size, out_channels, H, W]
        
        # 全局池化并计算通道注意力权重
        s = self.global_pool(U).view(batch_size, -1)  # [batch_size, out_channels]
        z = self.fc(s).view(batch_size, 3, self.out_channels, 1, 1)  # [batch_size, 3, out_channels, 1, 1]

        # 修改权重计算方式
        weights = z.split(1, dim=1)  # 按分支数分割
        output = sum(w.squeeze(1) * out for w, out in zip(weights, outs))  # [batch_size, out_channels, H, W]
        output += U  # 添加残差连接
        return output



class DINOv2FeatureProcessor(nn.Module):
    def __init__(self, in_channels=384, out_channels=[48, 96, 192, 384]):
        super(DINOv2FeatureProcessor, self).__init__()
        
        # Project DINOv2 features from 384 to specified channels
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0
            ) for out_channel in out_channels
        ])
        
        # Resize layers for upsampling/downsampling
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=8,
                stride=8,
                padding=0  # Upsample by 8x
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=4,
                stride=4,
                padding=0  # Upsample by 4x
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[2],
                out_channels=out_channels[2],
                kernel_size=2,
                stride=2,
                padding=0  # Upsample by 2x
            ),
            nn.Identity()  # No change (1x)
        ])

    def forward(self, dinov2_feats):
        # Project features to different channels
        # 分别处理每个DINOv2特征
        projected_feats = [proj(feat) for proj, feat in zip(self.projects, dinov2_feats)]
        
        # 应用resize操作
        resized_feats = [resize_layer(feat) for resize_layer, feat in zip(self.resize_layers, projected_feats)]
        
        return resized_feats

class FeatureFusionModule(nn.Module):
    def __init__(self, cnn_channels=[16, 24, 40, 112], dinov2_channels=[48, 96, 192, 384], out_channels=64):
        super(FeatureFusionModule, self).__init__()
        self.sk_convs = nn.ModuleList([
            SKConv(in_channels=cnn_ch + dinov2_ch, out_channels=out_channels)
            for cnn_ch, dinov2_ch in zip(cnn_channels, dinov2_channels)
        ])

    def forward(self, dinov2_feats,cnn_feats):
        # 按层 concat 并通过 SKConv 融合
        fused_feats = []
        for cnn_feat, dinov2_feat, sk_conv in zip(cnn_feats, dinov2_feats, self.sk_convs):
            dinov2_feat = F.interpolate(dinov2_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=True) # 把上采样过后跟CNN特征比较接近
            concat_feat = torch.cat([cnn_feat, dinov2_feat], dim=1)  # [B, cnn_ch+dinov2_ch, H, W]
            fused = sk_conv(concat_feat)  # [B, out_channels, H, W]
            fused_feats.append(fused)
        return fused_feats


class LayersFusion(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, bn=True):
        super().__init__()
        self.bn = bn

        # Convolution to process concatenated high and low features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                in_channels_high + in_channels_low,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not bn
            ),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_res_feat, low_res_feat):
        # Align high-resolution feature size to low-resolution feature size using interpolation
        high_res_feat = F.interpolate(high_res_feat, size=low_res_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate high and low features along the channel dimension
        fused_feat = torch.cat([high_res_feat, low_res_feat], dim=1)
        
        # Apply convolution to fused features
        output = self.fusion_conv(fused_feat)
        
        return output

class Classifier(nn.Module):
    def __init__(self, in_chan=64, n_class=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1)
        )
    
    def forward(self, x):
        return self.head(x)

class CDNet(nn.Module):
    def __init__(self, config, n_class=9):
        super().__init__()
        self.backbone1 = DINOv2("small")
        self.backbone2 = MobileNetV3Backbone(
            model_name=config['mobile_v3_version'],
            feature_layers=config['MobileNetv3_layer'],
            weights=config['mobile_v3_weights'],
            freeze_layers=None
        )
        self.intermediate_layer_id = [2, 5, 8, 11]
        self.dinov2_processor = DINOv2FeatureProcessor(out_channels=config['DINOV2_dim'])
        # [64, 64, 64, 64]
        self.Dino_CNN_fusion = FeatureFusionModule(cnn_channels=config['MobileNetv3_dim'],dinov2_channels=config['DINOV2_dim'])
        self.layer_fusions = nn.ModuleList([
            LayersFusion(in_channels_high=64, in_channels_low=64, out_channels=64),  # 用于融合不同层级的特征
            LayersFusion(in_channels_high=64, in_channels_low=64, out_channels=64),
            LayersFusion(in_channels_high=64, in_channels_low=64, out_channels=64)
        ])
        self.conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.change_info = ChangeInformationModel(config ,in_channels=64)
        self.memory = Memory(
            memory_size=n_class,
            feature_dim=64,
            key_dim=64,
            temp_update=config.get('temp_update', 0.1),
            temp_gather=config.get('temp_gather', 0.1),
            momentum=config.get('momentum', 0.8),
            num_prototypes=config.get('num_prototypes', 4),
            update_interval=config.get('update_interval', 10),
            topk_queries=config.get('topk_queries', 0.0001),
            max_skip_batches=config.get('max_skip_batches', 10)
        )
        self.classifier = Classifier(in_chan=64, n_class=n_class)

    def forward(self, img1, img2, labels=None):
        patch_h, patch_w = img1.shape[-2] // 14, img2.shape[-1] // 14
        feature1 = self.backbone1.get_intermediate_layers(img1, self.intermediate_layer_id, reshape=True)
        feature2 = self.backbone1.get_intermediate_layers(img2, self.intermediate_layer_id, reshape=True)
        cnn_x1 = self.backbone2(img1)
        cnn_x2 = self.backbone2(img2)

        #处理feature1,2
        feature1 = self.dinov2_processor(feature1)
        feature2 = self.dinov2_processor(feature2)

        
        x1s = self.Dino_CNN_fusion(feature1, cnn_x1)
        x2s = self.Dino_CNN_fusion(feature2, cnn_x2)
        
        #多层级融合
                # 多层级融合
        # 从高分辨率到低分辨率逐层融合
        x1 = x1s[-1]  # 从最高层开始
        for i in range(len(x1s)-2, -1, -1):
            x1 = self.layer_fusions[i](x1, x1s[i])
            
        x2 = x2s[-1]  # 从最高层开始
        for i in range(len(x2s)-2, -1, -1):
            x2 = self.layer_fusions[i](x2, x2s[i])

        # 使用差异模块替换concat
        # x = self.diff_module(x1, x2)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv(x)
        x = self.change_info(x1,x2)
        # x = self.diff_info(x1, x2)
        
        if self.training and labels is not None:
            if labels.dim() == 4:
                labels = torch.argmax(labels, dim=1)
            B, C, H, W = x.shape
            labels = F.interpolate(labels.float().unsqueeze(1), size=(H, W), mode='nearest').long().squeeze(1)
        else:
            labels = None
        
        updated_x = self.memory(x, labels=labels, train=self.training)
        x = self.classifier(updated_x)
        x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
        return x
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()