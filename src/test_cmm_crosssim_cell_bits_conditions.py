"""
批量构建不同 CMM-on-CrossSim cell_bits 下的条件模型。

默认扫描 cell_bits = 0, 2, 3, 4, 6, 8，其中 0 表示连续电导。固定
array=128x128、ADC/DAC=0/0、read_noise_std=1e-4、write_noise_std=1e-4。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认 cell_bits 扫描（0, 2, 3, 4, 6, 8）
#   python -m src.test_cmm_crosssim_cell_bits_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt
#
#   :: 自定义 bit 列表，跳过已存在的 checkpoint
#   python -m src.test_cmm_crosssim_cell_bits_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_crosssim_cell_bits_conditions ^
#       --cell-bits-list 0,4,8 --skip-existing
#
#   :: CPU 构建，仅映射 decoder layers
#   python -m src.test_cmm_crosssim_cell_bits_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --device cpu --scope layers_only

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

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
    """解析 CMM-on-CrossSim cell_bits sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CMM-on-CrossSim checkpoints for cell_bits sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_crosssim_cell_bits_conditions"),
        help="CMM-on-CrossSim cell_bits 条件模型输出目录",
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
    parser.add_argument("--tile-rows", type=int, default=128, help="CMM 阵列最大行数")
    parser.add_argument("--tile-cols", type=int, default=128, help="CMM 阵列最大列数")
    parser.add_argument("--rmin", type=float, default=1e3, help="Ron，器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="Roff，器件最大电阻")
    parser.add_argument("--adc-resolution", type=int, default=0, help="固定 ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=0, help="固定 DAC 分辨率；0 表示理想 DAC")
    parser.add_argument("--read-noise-std", type=float, default=1e-4, help="固定读噪声强度")
    parser.add_argument("--write-noise-std", type=float, default=1e-4, help="固定写入噪声强度")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证 programming_error 抽样可复现")
    parser.add_argument(
        "--cell-bits-list",
        type=str,
        default="0,2,3,4,6,8",
        help="逗号分隔的 CMM cell_bits 列表；0 表示连续电导",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def parse_cell_bits(raw: str) -> List[int]:
    """将命令行 cell_bits 列表解析为整数，并检查范围。"""
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--cell-bits-list 至少需要一个值。")
    if any(value < 0 for value in values):
        raise ValueError("cell_bits 不能为负数。")
    return values


def condition_name(cell_bits: int) -> str:
    """生成稳定的 cell_bits 条件名。"""
    return "cell-continuous" if cell_bits == 0 else f"cell-{cell_bits}bit"


def set_seed(seed: int) -> None:
    """设置随机种子，保证 CrossSim programming_error 抽样尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 cell_bits 逐个构建 CMM-on-CrossSim checkpoint。"""
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    # 固定本实验的非 sweep 变量，避免命令行误改导致实验含义漂移。
    args.tile_rows = 128
    args.tile_cols = 128
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
    for cell_bits in parse_cell_bits(args.cell_bits_list):
        set_seed(args.seed)
        args.cell_bits = cell_bits
        condition = condition_name(cell_bits)
        out_path = args.output_dir / f"caption_transformer_{condition}_cmm_crosssim.pt"

        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "cell_bits": str(cell_bits),
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} cell_bits={cell_bits}")
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
                "cell_bits": cell_bits,
                "seed": args.seed,
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
                "cell_bits": str(cell_bits),
                "tile_shape": "128x128",
                "adc_resolution": "0",
                "dac_resolution": "0",
                "write_noise_std": "0.0001",
                "crosssim_programming_error_std": "0.0001",
                "read_noise_std": "0.0001",
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
    """主入口：构建 CMM-on-CrossSim cell_bits 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM-on-CrossSim cell_bits condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: cell_bits={row['cell_bits']}, "
            f"num_mapped_linear={row.get('num_mapped_linear', 'unknown')} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
