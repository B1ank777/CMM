"""
图像描述生成模型 —— 编码器-解码器架构。
编码器使用 CNN（ResNet/VGG）提取图像特征，解码器使用 Transformer 自回归生成描述文本。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .decoder import DecoderLayer
from .encoder import ResNetEncoder


class CaptionTransformer(nn.Module):
    """图像描述生成 Transformer 模型。

    架构: CNN 编码器（图像 → 特征向量）+ Transformer 解码器（自回归生成文本）。
    支持 ResNet 编码器，训练时可选择冻结或微调 CNN 权重。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 2,
        max_len: int = 30,
        encoder_type: str = "resnet",
        pad_id: int = 0,
        dropout: float = 0.1,
        train_cnn: bool = False,
        pretrained_encoder: bool = True,
    ):
        """
        参数:
            vocab_size: 词表大小。
            d_model: 隐藏维度。
            num_heads: 注意力头数。
            ffn_dim: 前馈网络的中间维度。
            num_layers: 解码器层的数量。
            max_len: 最大描述长度。
            encoder_type: 编码器类型，支持 'resnet' 或 'vgg16'。
            pad_id: 填充 token 的 ID。
            dropout: dropout 比率。
            train_cnn: 是否训练 CNN 权重（True=微调，False=冻结）。
            pretrained_encoder: 是否使用预训练的 CNN 权重。
        """
        super().__init__()

        # 图像编码器：CNN 提取视觉特征
        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(
                d_model=d_model,
                train_cnn=train_cnn,
                pretrained=pretrained_encoder,
            )
        elif encoder_type == "vgg16":
            raise NotImplementedError("VGG16 编码器尚未实现。")
        else:
            raise ValueError("encoder_type 必须是 'vgg16' 或 'resnet'")

        # 文本嵌入 + 位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # 堆叠的解码器层
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(d_model, vocab_size)  # 投影到词表维度
        self.dropout = nn.Dropout(dropout)

        self.max_len = max_len
        self.d_model = d_model
        self.pad_id = pad_id

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """构建因果遮罩，确保解码时每个位置只能看到自身及之前的位置。

        参数:
            seq_len: 序列长度。
            device: 张量所在设备。

        返回:
            形状 [1, 1, seq_len, seq_len] 的布尔张量，True 表示可见。
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask[None, None, :, :]

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        参数:
            images: 输入图像，形状 [B, 3, 224, 224]。
            captions: 输入描述 token，形状 [B, T]。

        返回:
            logits，形状 [B, T, vocab_size]。
        """
        bsz, seq_len = captions.shape
        if seq_len > self.max_len:
            raise ValueError(f"描述长度 {seq_len} 超过最大长度 {self.max_len}。")

        device = captions.device

        # 图像编码：提取视觉特征 memory
        memory = self.encoder(images)

        # 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)

        # 词嵌入（Paper 风格缩放）+ 位置嵌入
        x = self.token_embedding(captions) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        # 因果遮罩 + 填充遮罩
        tgt_mask = self.make_causal_mask(seq_len, device)
        tgt_key_padding_mask = captions.eq(self.pad_id)

        # 逐层解码
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        logits = self.output_proj(x)  # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归生成图像描述。

        从 BOS token 开始，逐词预测直到遇到 EOS 或达到最大长度。

        参数:
            image: 输入图像，形状 [C, H, W] 或 [1, C, H, W]。
            bos_id: 起始 token ID（BOS）。
            eos_id: 结束 token ID（EOS）。
            max_len: 最大生成长度，默认使用模型配置的 max_len。

        返回:
            生成的 token 序列，形状 [T]。
        """
        self.eval()

        # 自动添加 batch 维度
        if image.dim() == 3:
            image = image.unsqueeze(0)

        max_len = max_len or self.max_len
        if max_len > self.max_len:
            raise ValueError(f"生成长度 {max_len} 超过模型最大长度 {self.max_len}。")

        device = image.device
        tokens = torch.tensor([[bos_id]], device=device, dtype=torch.long)

        # 逐词自回归生成
        for _ in range(max_len - 1):
            logits = self.forward(image, tokens)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 贪心解码
            tokens = torch.cat([tokens, next_id], dim=1)
            if int(next_id.item()) == eos_id:
                break

        return tokens.squeeze(0)
