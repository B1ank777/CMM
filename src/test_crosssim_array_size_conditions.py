"""
批量构建不同 CrossSim 交叉阵列规模下的条件模型。

默认只改变阵列最大行列数（tile_rows/tile_cols），ADC/DAC 使用中等非理想基线，
用于观察 64x64 / 128x128 / 256x256 / 512x512 分块规模对模型保真度的影响。
如需验证纯理想 tile 等价性，请显式传入 --adc-resolution 0 --dac-resolution 0。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认方阵尺寸扫描（64, 128, 256, 512），中等非理想 ADC/DAC
#   python -m src.test_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_array_size_conditions
#
#   :: 自定义方阵尺寸，跳过已存在的文件
#   python -m src.test_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_array_size_conditions ^
#       --array-sizes 32,64,128 --skip-existing
#
#   :: 矩形阵列尺寸，GPU 加速
#   python -m src.test_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_array_rect ^
#       --device cuda --use-gpu --rect-array-sizes 64x128,128x256,256x512
#
#   :: 理想 ADC/DAC，仅验证 tile 分块本身的影响
#   python -m src.test_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_array_ideal ^
#       --adc-resolution 0 --dac-resolution 0 --array-sizes 32,64,128,256,512

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .map_crosssim import (
    build_crosssim_model,
    build_model_from_payload,
    freeze_encoder,
    load_checkpoint,
    make_crosssim_args,
    should_use_crosssim_gpu,
)


def parse_args() -> argparse.Namespace:
    """解析 array size sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CrossSim checkpoints for array-size sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/crosssim_array_size_conditions"),
        help="array size 条件模型输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="映射与推理设备；cuda 会启用 CrossSim GPU 后端",
    )
    parser.add_argument("--use-gpu", action="store_true", help="显式要求 CrossSim 使用 GPU")
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="CrossSim 映射范围",
    )
    parser.add_argument("--adc-resolution", type=int, default=10, help="ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=12, help="DAC 分辨率；0 表示理想 DAC")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数")
    parser.add_argument("--rmin", type=float, default=1e3, help="器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="读噪声强度；0 表示关闭")
    parser.add_argument("--programming-error-std", type=float, default=0.0, help="CrossSim 内部写入误差；0 表示关闭")
    parser.add_argument(
        "--array-sizes",
        type=str,
        default="64,128,256,512",
        help="逗号分隔的方阵尺寸列表，例如 64,128,256,512",
    )
    parser.add_argument(
        "--rect-array-sizes",
        type=str,
        default="",
        help="可选矩形尺寸列表，例如 64x128,128x256；若提供则覆盖 --array-sizes",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def parse_array_sizes(args: argparse.Namespace) -> List[Tuple[int, int]]:
    """解析方阵或矩形阵列尺寸列表。"""
    if args.rect_array_sizes.strip():
        sizes: List[Tuple[int, int]] = []
        for item in args.rect_array_sizes.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if "x" not in item:
                raise ValueError("--rect-array-sizes 项必须形如 64x128。")
            rows, cols = item.split("x", maxsplit=1)
            sizes.append((int(rows), int(cols)))
    else:
        sizes = [(int(item.strip()), int(item.strip())) for item in args.array_sizes.split(",") if item.strip()]

    if not sizes:
        raise ValueError("至少需要一个 array size。")
    if any(rows <= 0 or cols <= 0 for rows, cols in sizes):
        raise ValueError("array size 的行列数必须为正整数。")
    return sizes


def condition_name(tile_rows: int, tile_cols: int) -> str:
    """生成稳定的 array size 条件名。"""
    return f"array-{tile_rows}x{tile_cols}"


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按阵列规模逐个构建 CrossSim checkpoint。"""
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for tile_rows, tile_cols in parse_array_sizes(args):
        # make_crosssim_args 需要当前单个 tile_rows/tile_cols。
        args.tile_rows = tile_rows
        args.tile_cols = tile_cols
        condition = condition_name(tile_rows, tile_cols)
        out_path = args.output_dir / f"caption_transformer_{condition}_crosssim.pt"

        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "tile_rows": str(tile_rows),
                    "tile_cols": str(tile_cols),
                    "array_size": f"{tile_rows}x{tile_cols}",
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} tile_shape=({tile_rows}, {tile_cols})")
        t0 = time.time()

        crosssim_model = build_crosssim_model(
            model=base_model,
            scope=args.scope,
            tile_shape=(tile_rows, tile_cols),
            adc_resolution=args.adc_resolution,
            dac_resolution=args.dac_resolution,
            bias_rows=args.bias_rows,
            use_gpu=use_gpu,
            rmin=args.rmin,
            rmax=args.rmax,
            cell_bits=args.cell_bits,
            read_noise_std=args.read_noise_std,
            programming_error_std=args.programming_error_std,
        )
        crosssim_model.to(device)
        crosssim_model.eval()

        torch.save(
            {
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
                "array_size": f"{tile_rows}x{tile_cols}",
                "model_config": payload.get("model_config"),
                "vocab_stoi": payload.get("vocab_stoi"),
                "crosssim_model_state_dict": crosssim_model.state_dict(),
                "crosssim_args": make_crosssim_args(args, use_gpu),
            },
            out_path,
        )

        rows.append(
            {
                "condition": condition,
                "tile_rows": str(tile_rows),
                "tile_cols": str(tile_cols),
                "array_size": f"{tile_rows}x{tile_cols}",
                "checkpoint": str(out_path),
            }
        )
        print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

        del crosssim_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = args.output_dir / "conditions_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")
    return rows


def main() -> None:
    """主入口：构建 array size 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CrossSim array-size condition models ===")
    for row in rows:
        print(f"{row['condition']}: array_size={row['array_size']} -> {row['checkpoint']}")


if __name__ == "__main__":
    main()
