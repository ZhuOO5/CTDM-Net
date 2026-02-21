import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Backbone(nn.Module):
    def __init__(self, model_name='mobilenet_v3_large', feature_layers=[2, 4, 7, 13], weights='IMAGENET1K_V1', freeze_layers=None):
        super(MobileNetV3Backbone, self).__init__()
        # 加载预训练的 MobileNetV3 模型（支持 Small 或 Large 版本）
        if model_name not in ['mobilenet_v3_small', 'mobilenet_v3_large']:
            raise ValueError("model_name must be 'mobilenet_v3_small' or 'mobilenet_v3_large'")
        self.model = getattr(models, model_name)(weights=weights)
        
        # 获取特征提取部分（卷积和 bneck 层）
        self.features = self.model.features
        
        # 验证层数是否有效
        max_layer = len(self.features) - 1
        for layer in feature_layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(f"Feature layer {layer} is out of range [0, {max_layer}]")
        self.feature_layers = sorted(feature_layers)  # 确保层按顺序提取
        
        # 确保参数可训练（默认 requires_grad=True）
        for param in self.features.parameters():
            param.requires_grad = True
            
        # 可选：冻结指定层（例如浅层），如果 freeze_layers 提供了层编号
        if freeze_layers is not None:
            for i, layer in enumerate(self.features):
                if i in freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
    # def forward(self, x):
    #     """
    #     前向传播：输入 x 经过整个 MobileNetV3 的特征提取层，
    #     仅在指定层（feature_layers）收集输出特征图。
    #     所有未冻结的参数支持反向传播，用于微调。
        
    #     Args:
    #         x: 输入张量，形状 [B, 3, H, W]（例如 [B, 3, 224, 224]）
        
    #     Returns:
    #         features: 列表，包含指定层的特征图，形状为 [B, C_i, H_i, W_i]
    #     """
    #     features = []
    #     # 逐层处理整个特征提取部分
    #     for i, layer in enumerate(self.features):
    #         x = layer(x)  # 更新 x 为当前层的输出
    #         if i in self.feature_layers:
    #             # 保存指定层的特征图
    #             features.append(x.clone())  # 使用 clone() 确保不修改原始张量
    #     return features
    def forward(self, x):
        """
        前向传播：输入 x 经过 EfficientNet 的特征提取层，
        仅在指定层（feature_layers）收集输出特征图，并在最后一个指定层后停止。
        所有未冻结的参数支持反向传播，用于微调。
        
        Args:
            x: 输入张量，形状 [B, 3, H, W]（例如 [B, 3, 224, 224]）
        
        Returns:
            features: 列表，包含指定层的特征图，形状为 [B, C_i, H_i, W_i]
        """
        features = []
        last_layer = max(self.feature_layers)  # 获取最后一个需要提取的层
        for i, layer in enumerate(self.features):
            if i > last_layer:  # 在最后一个指定层后停止
                break
            x = layer(x)  # 更新 x 为当前层的输出
            if i in self.feature_layers:
                # 保存指定层的特征图
                features.append(x.clone())  # 使用 clone() 确保不修改原始张量
        return features

# # 使用示例
# if __name__ == "__main__":
#     # 输入示例
#     batch_size = 4
#     x = torch.randn(batch_size, 3, 224, 224)
    
#     # 初始化 backbone（默认所有层可训练）
#     backbone = MobileNetV3Backbone(
#         model_name='mobilenet_v3_large',
#         feature_layers=[2, 4, 7, 13],
#         pretrained=True,
#         freeze_layers=None  # 可设置为 [0, 1, 2] 冻结浅层
#     )
    
#     # 前向传播
#     features = backbone(x)
    
#     # 输出特征维度
#     for i, feat in enumerate(features):
#         print(f"Feature layer {backbone.feature_layers[i]} shape: {feat.shape}")
    
#     # 验证参数是否可训练
#     for name, param in backbone.named_parameters():
#         print(f"Parameter {name}: requires_grad = {param.requires_grad}")