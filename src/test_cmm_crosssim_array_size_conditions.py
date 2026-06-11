"""
批量构建不同 CMM-on-CrossSim array size 下的条件模型。

默认只改变 tile_rows/tile_cols，扫描 64x64 / 128x128 / 256x256 / 512x512。
固定 ADC/DAC=0/0、cell_bits=0、read_noise_std=1e-4、write_noise_std=1e-4。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认方阵 array size 扫描（64, 128, 256, 512）
#   python -m src.test_cmm_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt
#
#   :: 自定义方阵尺寸，跳过已存在的 checkpoint
#   python -m src.test_cmm_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_crosssim_array_size_conditions ^
#       --array-sizes 32,64,128 --skip-existing
#
#   :: 矩形 array size
#   python -m src.test_cmm_crosssim_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_crosssim_array_rect ^
#       --rect-array-sizes 64x128,128x256,256x512

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .map_cmm_crosssim import build_cmm_crosssim_model, make_cmm_crosssim_args
from .map_crosssim import (
    build_model_from_payload,
    count_linear_layers_by_scope,
    freeze_encoder,
    load_checkpoint,
    make_crosssim_args,
    should_use_crosssim_gpu,
)


def parse_args() -> argparse.Namespace:
    """解析 CMM-on-CrossSim array size sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CMM-on-CrossSim checkpoints for array-size sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_crosssim_array_size_conditions"),
        help="CMM-on-CrossSim array size 条件模型输出目录",
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
        help="CMM-on-CrossSim 映射范围",
    )
    parser.add_argument("--rmin", type=float, default=1e3, help="Ron，器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="Roff，器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="固定单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--adc-resolution", type=int, default=0, help="固定 ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=0, help="固定 DAC 分辨率；0 表示理想 DAC")
    parser.add_argument("--read-noise-std", type=float, default=1e-4, help="固定读噪声强度")
    parser.add_argument("--write-noise-std", type=float, default=1e-4, help="固定写入噪声强度")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证 programming_error 抽样可复现")
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
    """解析方阵或矩形 array size 列表。"""
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


def set_seed(seed: int) -> None:
    """设置随机种子，保证 CrossSim programming_error 抽样尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 array size 逐个构建 CMM-on-CrossSim checkpoint。"""
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    # 固定本实验的非 sweep 变量，避免命令行误改导致实验含义漂移。
    args.cell_bits = 0
    args.adc_resolution = 0
    args.dac_resolution = 0
    args.read_noise_std = 1e-4
    args.write_noise_std = 1e-4
    args.programming_error_std = args.write_noise_std

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for tile_rows, tile_cols in parse_array_sizes(args):
        set_seed(args.seed)
        args.tile_rows = tile_rows
        args.tile_cols = tile_cols
        condition = condition_name(tile_rows, tile_cols)
        out_path = args.output_dir / f"caption_transformer_{condition}_cmm_crosssim.pt"

        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "tile_rows": str(tile_rows),
                    "tile_cols": str(tile_cols),
                    "array_size": f"{tile_rows}x{tile_cols}",
                    "adc_resolution": str(args.adc_resolution),
                    "dac_resolution": str(args.dac_resolution),
                    "cell_bits": str(args.cell_bits),
                    "write_noise_std": str(args.write_noise_std),
                    "crosssim_programming_error_std": str(args.write_noise_std),
                    "read_noise_std": str(args.read_noise_std),
                    "seed": str(args.seed),
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} tile_shape=({tile_rows}, {tile_cols})")
        t0 = time.time()
        num_mapped_linear = count_linear_layers_by_scope(base_model, args.scope)

        crosssim_model = build_cmm_crosssim_model(base_model, args, use_gpu)
        crosssim_model.to(device)
        crosssim_model.eval()

        torch.save(
            {
                "format": "cmm_crosssim_v1",
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
                "array_size": f"{tile_rows}x{tile_cols}",
                "seed": args.seed,
                "cell_bits": args.cell_bits,
                "adc_resolution": args.adc_resolution,
                "dac_resolution": args.dac_resolution,
                "read_noise_std": args.read_noise_std,
                "write_noise_std": args.write_noise_std,
                "crosssim_programming_error_std": args.write_noise_std,
                "model_config": payload.get("model_config"),
                "vocab_stoi": payload.get("vocab_stoi"),
                "crosssim_model_state_dict": crosssim_model.state_dict(),
                "crosssim_args": make_crosssim_args(args, use_gpu),
                "cmm_crosssim_args": make_cmm_crosssim_args(args, use_gpu),
                "num_mapped_linear": num_mapped_linear,
            },
            out_path,
        )

        rows.append(
            {
                "condition": condition,
                "tile_rows": str(tile_rows),
                "tile_cols": str(tile_cols),
                "array_size": f"{tile_rows}x{tile_cols}",
                "adc_resolution": str(args.adc_resolution),
                "dac_resolution": str(args.dac_resolution),
                "cell_bits": str(args.cell_bits),
                "write_noise_std": str(args.write_noise_std),
                "crosssim_programming_error_std": str(args.write_noise_std),
                "read_noise_std": str(args.read_noise_std),
                "seed": str(args.seed),
                "num_mapped_linear": str(num_mapped_linear),
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
    """主入口：构建 CMM-on-CrossSim array size 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM-on-CrossSim array-size condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: array_size={row['array_size']}, "
            f"num_mapped_linear={row.get('num_mapped_linear', 'unknown')} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
