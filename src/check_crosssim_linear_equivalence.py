from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .map_crosssim import build_crosssim_params, should_use_crosssim_gpu, synchronize_crosssim_cores


def parse_args() -> argparse.Namespace:
    """解析 CrossSim 单层等价性检查参数。"""
    parser = argparse.ArgumentParser(
        description="Check CrossSim AnalogLinear equivalence against torch.nn.Linear."
    )
    parser.add_argument("--device", type=str, default="cpu", help="检查设备；cuda 会启用 CrossSim GPU")
    parser.add_argument("--use-gpu", action="store_true", help="显式要求 CrossSim 使用 GPU")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=8)
    parser.add_argument("--out-features", type=int, default=4)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--tile-cols", type=int, default=16)
    parser.add_argument("--bias-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """构造 nn.Linear 与 CrossSim AnalogLinear，比较 3D 输入下的输出误差。"""
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    from simulator.algorithms.dnn.torch import from_torch

    params = build_crosssim_params(
        tile_shape=(args.tile_rows, args.tile_cols),
        adc_resolution=0,
        dac_resolution=0,
        use_gpu=use_gpu,
    )

    digital = nn.Linear(args.in_features, args.out_features).to(device).eval()
    analog = from_torch(digital, params, bias_rows=args.bias_rows).to(device).eval()
    synchronize_crosssim_cores(analog)

    # Transformer decoder 的线性层常见输入是 [B, T, D]，这里专门覆盖 3D 输入。
    x = torch.randn(args.batch_size, args.seq_len, args.in_features, device=device)
    digital_y = digital(x)
    analog_y = analog(x)

    diff = (digital_y - analog_y).abs()
    print("=== CrossSim Linear Equivalence Check ===")
    print(f"Device: {device}")
    print(f"CrossSim GPU: {use_gpu}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(analog_y.shape)}")
    print(f"MAE: {diff.mean().item():.10f}")
    print(f"Max error: {diff.max().item():.10f}")


if __name__ == "__main__":
    main()
