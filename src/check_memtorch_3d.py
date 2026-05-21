from __future__ import annotations

import argparse
import copy
from typing import Tuple

import torch
import torch.nn as nn

from .map_memtorch import patch_module_memristive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether patched MemTorch Linear handles 3D Transformer inputs correctly."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=12)
    parser.add_argument("--tile-rows", type=int, default=8)
    parser.add_argument("--tile-cols", type=int, default=8)
    parser.add_argument("--adc-resolution", type=int, default=8)
    parser.add_argument("--max-input-voltage", type=float, default=0.3)
    parser.add_argument("--ron", type=float, default=1e2)
    parser.add_argument("--roff", type=float, default=1e4)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def find_memtorch_linear(module: nn.Module) -> nn.Module:
    for child in module.modules():
        if type(child).__name__ == "Linear" and type(child).__module__.startswith("memtorch"):
            return child
    raise RuntimeError("No MemTorch Linear module found after patching.")


@torch.no_grad()
def compare_3d_vs_flat(
    mem_linear: nn.Module,
    x3d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """比较 3D 输入路径与手动 flatten 输入路径是否一致。

    正确行为应该是:
        MemLinear([B, T, D]) == MemLinear([B*T, D]).reshape(B, T, -1)
    """
    y3d = mem_linear(x3d)
    flat_input = x3d.reshape(-1, x3d.shape[-1])
    flat_output = mem_linear(flat_input)
    yflat = flat_output.reshape(*x3d.shape[:-1], flat_output.shape[-1])
    diff = (y3d - yflat).abs()
    return y3d, yflat, diff


@torch.no_grad()
def compare_against_digital(
    digital_linear: nn.Linear,
    x3d: torch.Tensor,
    mem_output: torch.Tensor,
) -> torch.Tensor:
    """额外观察 MemTorch 结果相对原始 nn.Linear 的误差量级。"""
    # 某些 MemTorch 后端即使输入在 CPU，也会把输出放到 CUDA；这里显式对齐 device。
    digital_output = digital_linear(x3d).to(mem_output.device)
    return (digital_output - mem_output).abs()


def run_case(
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)

    digital = nn.Linear(in_features, out_features).eval()
    container = nn.Sequential(copy.deepcopy(digital)).eval()

    # 使用和正式 mapping 相同的函数，确保检查覆盖当前生产路径里的 3D reshape 修补逻辑。
    mem_container = patch_module_memristive(
        container,
        use_bindings=False,
        tile_shape=(args.tile_rows, args.tile_cols),
        max_input_voltage=args.max_input_voltage,
        adc_resolution=args.adc_resolution,
        ron=args.ron,
        roff=args.roff,
    ).to(device).eval()

    digital = digital.to(device)
    mem_linear = find_memtorch_linear(mem_container)
    x3d = torch.randn(batch_size, seq_len, in_features, device=device)

    y3d, yflat, shape_diff = compare_3d_vs_flat(mem_linear, x3d)
    digital_diff = compare_against_digital(digital, x3d, y3d)

    print(f"--- Case B={batch_size}, T={seq_len}, D_in={in_features}, D_out={out_features} ---")
    print(f"Input shape: {tuple(x3d.shape)}")
    print(f"Output device: {y3d.device}")
    print(f"3D output shape: {tuple(y3d.shape)}")
    print(f"Flat+reshape output shape: {tuple(yflat.shape)}")
    print(f"3D vs flat MAE: {shape_diff.mean().item():.10f}")
    print(f"3D vs flat max error: {shape_diff.max().item():.10f}")
    print(f"Digital vs MemTorch MAE: {digital_diff.mean().item():.10f}")
    print(f"Digital vs MemTorch max error: {digital_diff.max().item():.10f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=== MemTorch 3D Linear Check ===")
    print(f"Device: {args.device}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    print(f"ADC resolution: {args.adc_resolution}")

    run_case(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        in_features=args.in_features,
        out_features=args.out_features,
        args=args,
    )
    # 单样本生成 caption 时最容易触发 batch_size=1 的边界问题，必须单独检查。
    run_case(
        batch_size=1,
        seq_len=args.seq_len,
        in_features=args.in_features,
        out_features=args.out_features,
        args=args,
    )


if __name__ == "__main__":
    main()
