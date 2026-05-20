from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    """图像编码器：用 ResNet-50 将图像编码为视觉 Token 序列，供交叉注意力记忆模块使用"""

    def __init__(
        self,
        d_model: int = 512,
        train_cnn: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()

        # 加载预训练权重（IMAGENET1K_V2），离线环境下回退到随机初始化
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            resnet = models.resnet50(weights=weights)
        except Exception:
            resnet = models.resnet50(weights=None)

        # 去掉最后两层（全局平均池化和全连接层），保留卷积特征图输出
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # 线性投影：将 2048 维特征映射到 d_model 维
        self.proj = nn.Linear(2048, d_model)

        # 冻结 CNN 主干网络参数
        if not train_cnn:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入图像 [B, 3, 224, 224]，输出视觉 Token 序列 [B, 49, d_model]"""
        x = self.backbone(images)        # 卷积特征图 [B, 2048, 7, 7]
        x = x.flatten(2).transpose(1, 2) # 展平空间维度 → [B, 49, 2048]
        x = self.proj(x)                 # 线性投影 → [B, 49, d_model]
        return x
