
import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import SelfAttention
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积：每个输入通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels,  # 输出通道数等于输入通道数
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation,
            groups=in_channels  # groups=in_channels 实现逐通道卷积
        )
        # 逐点卷积：1x1 卷积调整通道数
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
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

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        attn = self.fc(avg_out).view(b, c, 1, 1)
        attn = self.sigmoid(attn)
        return x * attn

class MultiBranchConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiBranchConv, self).__init__()
        self.branch1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=2, dilation=2) 
        self.branch3 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat((b1, b2, b3), dim=1)
        out = self.fusion(out)
        return out

class DownsampleSelfAttention(nn.Module):
    def __init__(self,nums_heads, channels, reduction=4):
        super(DownsampleSelfAttention, self).__init__()
        self.reduction = reduction
        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=reduction, padding=1)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=nums_heads,  # 可以根据需要调整头数
            dropout=0.1,
            batch_first=True
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)


    def forward(self, x):
        b, c, h, w = x.size()
        # 下采样
        x_down = self.downsample(x)  # [B, C, H//r, W//r]
        
        # 准备多头注意力输入
        x_down_flat = x_down.view(b, c, -1).permute(0, 2, 1)  # [B, (H*W)//(r^2), C]
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(
            query=x_down_flat,
            key=x_down_flat,
            value=x_down_flat
        )
        
        # 恢复形状 - calculate exact output dimensions
        new_h = (h + self.reduction - 1) // self.reduction  # 向上取整
        new_w = (w + self.reduction - 1) // self.reduction  # 向上取整
        out = attn_output.permute(0, 2, 1).reshape(b, c, new_h, new_w)
        # 使用插值恢复到原始尺寸
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        out = self.proj(out)
        return out
class PerformerSelfAttentionWrapper(nn.Module):
    def __init__(self, nums_heads, channels, dropout, causal, nb_features, kernel_ratio, redraw_interval, generalized_attention, kernel_fn, qkv_bias, attn_out_bias):
        super(PerformerSelfAttentionWrapper, self).__init__()
        self.channels = channels
        self.nums_heads = nums_heads

        # Performer SelfAttention with tunable parameters
        self.attn = SelfAttention(
            dim=channels,
            heads=nums_heads,
            # dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            nb_features=nb_features,
            # kernel_ratio=kernel_ratio,
            feature_redraw_interval=redraw_interval,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qkv_bias=qkv_bias,
            attn_out_bias=attn_out_bias
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        seq_len = h * w

        # Flatten for attention: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.view(b, c, seq_len).permute(0, 2, 1)  # [B, H*W, C]

        # Apply Performer attention
        out = self.attn(x_flat)  # [B, H*W, C]

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = out.permute(0, 2, 1).reshape(b, c, h, w)  # [B, C, H, W]
        out = self.proj(out)
        return out
class ChangeInformationModel(nn.Module):
    def __init__(self, config, in_channels=512):
        super(ChangeInformationModel, self).__init__()
        self.diff_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.concat_branch = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.spatial_att = SpatialAttention(in_channels)
        self.channel_att = ChannelAttention(in_channels)
        self.diff_conv = MultiBranchConv(in_channels, in_channels)

        # 处理kernel_fn参数
        kernel_fn = config.get('kernel_fn', 'relu')
        if isinstance(kernel_fn, str):
            kernel_fn = kernel_fn.lower()
            if kernel_fn == "relu":
                kernel_fn = nn.ReLU()
            elif kernel_fn == "tanh":
                kernel_fn = nn.Tanh()
            elif kernel_fn == "softplus":
                kernel_fn = nn.Softplus()
            else:
                raise ValueError(f"Unknown kernel function: {kernel_fn}")

        self.concat_att = PerformerSelfAttentionWrapper(
            nums_heads=config['dif_num_heads'], # 4,8
            channels=in_channels,
            dropout=config.get('attn_dropout', 0.1), # 0.1, 0.2
            causal=config.get('causal', False), # 图像任务更适合Casual
            nb_features=config.get('nb_features', 64), # nb_features 是 Performer 的随机特征核方法中使用的随机特征数量，设置为None就自动算
            kernel_ratio=config.get('kernel_ratio', 0.5), # Performer 使用随机特征核方法近似注意力，kernel_ratio 决定随机特征数量相对于序列长度（seq_len = H*W）的缩放比；
            redraw_interval=config.get('redraw_interval', 100), # 对于图像任务，redraw_interval 的影响通常较小，100 是一个合理的默认值。
            generalized_attention=config.get('generalized_attention', False), # 图像变化检测需要捕捉像素间的复杂关系，generalized_attention=True 可能帮助模型处理非线性模式，但需实验验证。
            kernel_fn=kernel_fn ,
            qkv_bias=config.get('qkv_bias', False),
            attn_out_bias=config.get('attn_out_bias', True)
        )
        self.gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.residual = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        concat = torch.cat((x1, x2), dim=1)

        diff_out = self.diff_branch(diff)
        concat_out = self.concat_branch(concat)

        spatial_guidance = self.spatial_att(diff_out)
        concat_out = concat_out + self.channel_att(concat_out) * spatial_guidance

        diff_out = diff_out + self.channel_att(concat_out) * diff_out

        diff_out = self.diff_conv(diff_out)
        concat_out = self.concat_att(concat_out)

        gate = torch.sigmoid(self.gate(torch.cat((diff_out, concat_out), dim=1)))
        fused = gate * diff_out + (1 - gate) * concat_out
        out = self.final_conv(fused) + self.residual(torch.cat((diff_out, concat_out), dim=1))
        return out