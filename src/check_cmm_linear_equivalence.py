from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .cmm import CMMLinear, count_cmm_linear_layers
from .map_cmm import build_cmm_model
from .map_crosssim import count_linear_layers_by_scope
from .models.decoder import DecoderLayer

# 运行示例：
#   # 默认参数验证 CMMLinear 等价性及 decoder_only 映射范围
#   python -m src.check_cmm_linear_equivalence
#
#   # 自定义设备和小模型参数
#   python -m src.check_cmm_linear_equivalence --device cpu --d-model 16 --num-layers 2


def parse_args() -> argparse.Namespace:
    """解析 CMM 等价性检查参数。"""
    parser = argparse.ArgumentParser(description="Check CMMLinear equivalence against torch.nn.Linear.")
    parser.add_argument("--device", type=str, default="cpu", help="检查设备")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=8)
    parser.add_argument("--out-features", type=int, default=4)
    parser.add_argument("--tile-rows", type=int, default=3)
    parser.add_argument("--tile-cols", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-5)
    return parser.parse_args()


class DummyCaptionLike(nn.Module):
    """只包含映射所需字段的轻量模型，避免检查脚本依赖 CNN 编码器。

    使用 DummyCaptionLike 替代完整的 CaptionTransformer，
    这样等价性检查不需要加载 ResNet-50 或 COCO 数据集。
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim, dropout=0.0)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)


@torch.no_grad()
def check_linear_equivalence(args: argparse.Namespace, device: torch.device) -> None:
    """验证 CMMLinear 在无噪声、连续状态下等价于 nn.Linear。

    分别测试 [B, D] 和 [B, T, D] 两种输入形状，
    确保单层 tile 分块前向后 MAE ≈ 0 且 max error 在容差范围内。
    """
    # 创建等价的 nn.Linear 和 CMMLinear
    digital = nn.Linear(args.in_features, args.out_features).to(device).eval()
    cmm = CMMLinear.from_linear(
        digital,
        read_noise_std=0.0,
        write_noise_std=0.0,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
    ).to(device).eval()

    # 测试 2D（batch×features）和 3D（batch×seq×features）输入
    x2d = torch.randn(args.batch_size, args.in_features, device=device)
    x3d = torch.randn(args.batch_size, args.seq_len, args.in_features, device=device)

    for label, x in [("2D", x2d), ("3D", x3d)]:
        digital_y = digital(x)
        cmm_y = cmm(x)
        diff = (digital_y - cmm_y).abs()
        mae = diff.mean().item()
        max_error = diff.max().item()
        print(f"{label} MAE: {mae:.10f}")
        print(f"{label} Max error: {max_error:.10f}")
        if max_error > args.tol:
            raise AssertionError(f"{label} CMMLinear max error {max_error:.6g} exceeds tol {args.tol:.6g}")


def check_decoder_only_mapping(args: argparse.Namespace, device: torch.device) -> None:
    """验证 decoder_only 映射包含 decoder layers 和 output_proj，且不替换 encoder。

    对比映射前的 nn.Linear 数量和映射后的 CMMLinear 数量，
    同时确保 encoder 中的 Linear 层未被替换。
    """
    model = DummyCaptionLike(
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
    ).to(device)
    expected = count_linear_layers_by_scope(model, "decoder_only")
    mapped = build_cmm_model(model, scope="decoder_only").to(device)
    actual = count_cmm_linear_layers(mapped)
    encoder_cmm = count_cmm_linear_layers(mapped.encoder)

    print(f"Expected decoder_only linear layers: {expected}")
    print(f"Mapped CMMLinear layers: {actual}")
    print(f"Encoder CMMLinear layers: {encoder_cmm}")

    if actual != expected:
        raise AssertionError(f"decoder_only mapped {actual} layers, expected {expected}.")
    if encoder_cmm != 0:
        raise AssertionError("decoder_only should not replace encoder linear layers.")


def main() -> None:
    """主入口：执行单层等价性检查和 decoder_only 映射范围检查。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 第一步：检查单层 CMMLinear ↔ nn.Linear 等价性
    print("=== CMM Linear Equivalence Check ===")
    print(f"Device: {device}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    check_linear_equivalence(args, device)

    # 第二步：检查 decoder_only 映射范围的正确性
    print("=== CMM Mapping Scope Check ===")
    check_decoder_only_mapping(args, device)


if __name__ == "__main__":
    main()
