"""
Transformer 解码器模块，包含多头注意力、解码器层和完整的 Transformer 解码器。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头注意力机制。

    将 Q/K/V 分别投影到多个头，在每个头上独立计算缩放点积注意力，
    最后将所有头的输出拼接并投影回原始维度。
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        """
        参数:
            d_model: 模型的隐藏维度，必须能被 num_heads 整除。
            num_heads: 注意力头的数量。
            dropout: 注意力权重上的 dropout 比率。
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度

        # Q/K/V/O 的线性投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            query: 查询张量，形状 [B, Tq, d_model]。
            key: 键张量，形状 [B, Tk, d_model]。
            value: 值张量，形状 [B, Tk, d_model]。
            attn_mask: 注意力遮罩，True/非0 表示可见位置。
            key_padding_mask: 键的填充遮罩，形状 [B, Tk]，True 表示填充位置。

        返回:
            注意力输出，形状 [B, Tq, d_model]。
        """
        bsz, tq, _ = query.shape
        tk = key.shape[1]

        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 拆分为多头: [B, num_heads, T, head_dim]
        q = q.view(bsz, tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tk, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力分数: [B, num_heads, Tq, Tk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用注意力遮罩（如因果遮罩），将不可见位置置为 -inf
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask = attn_mask[None, None, :, :]    # [1, 1, Tq, Tk]
            elif attn_mask.dim() == 3:
                mask = attn_mask[:, None, :, :]        # [B, 1, Tq, Tk]
            else:
                mask = attn_mask

            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores.masked_fill(mask == 0, float("-inf"))

        # 将填充位置置为 -inf
        if key_padding_mask is not None:
            # key_padding_mask: [B, Tk], True 表示填充位置
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        # Softmax + Dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和并合并多头
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, tq, self.d_model)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    """单层 Transformer 解码器层。

    结构: 自注意力 -> 交叉注意力 -> 前馈网络。
    每层均使用残差连接和 LayerNorm（Post-LN 风格）。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """
        参数:
            d_model: 隐藏维度。
            num_heads: 注意力头数。
            ffn_dim: 前馈网络的中间维度。
            dropout: dropout 比率。
        """
        super().__init__()

        # 带因果遮罩的自注意力
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        # 以编码器输出为 memory 的交叉注意力
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # 两层前馈网络: d_model -> ffn_dim -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            x: 解码器输入，形状 [B, T, d_model]。
            memory: 编码器输出，形状 [B, T_enc, d_model]。
            tgt_mask: 自注意力的因果遮罩。
            tgt_key_padding_mask: 目标序列的填充遮罩。

        返回:
            解码器层输出，形状 [B, T, d_model]。
        """
        # 自注意力 + 残差 + LayerNorm
        self_attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm1(x + self.dropout(self_attn_out))

        # 交叉注意力 + 残差 + LayerNorm
        cross_attn_out = self.cross_attn(query=x, key=memory, value=memory)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 前馈网络 + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class TransformerDecoder(nn.Module):
    """Transformer 解码器。

    由词嵌入、位置嵌入、堆叠的解码器层和输出投影层组成，
    支持自回归生成（通过因果遮罩确保位置 i 只能看到 i 及之前的位置）。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ):
        """
        参数:
            vocab_size: 词表大小。
            d_model: 隐藏维度。
            num_layers: 解码器层的数量。
            num_heads: 注意力头数。
            ffn_dim: 前馈网络的中间维度。
            dropout: dropout 比率。
            max_len: 位置嵌入支持的最大序列长度。
            pad_id: 填充 token 的 ID。
        """
        super().__init__()

        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

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
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)  # 语言模型输出头

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """构建因果遮罩（下三角矩阵），确保每个位置只能关注自身及之前的位置。

        参数:
            seq_len: 序列长度。
            device: 张量所在设备。

        返回:
            形状 [seq_len, seq_len] 的布尔张量，True 表示可见。
        """
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            input_ids: 输入 token ID，形状 [B, seq_len]。
            memory: 编码器输出，形状 [B, T_enc, d_model]。
            tgt_key_padding_mask: 目标填充遮罩，形状 [B, seq_len]，True 表示填充。

        返回:
            logits，形状 [B, seq_len, vocab_size]。
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"序列长度 {seq_len} 超过了最大长度 {self.pos_emb.num_embeddings}。"
            )

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # 从 input_ids 中自动生成填充遮罩
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = input_ids.eq(self.pad_id)

        # 构建因果遮罩（下三角矩阵）
        tgt_mask = self._build_causal_mask(seq_len, input_ids.device)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        x = self.norm(x)
        logits = self.lm_head(x)  # 投影到词表维度
        return logits
