"""
批量构建不同 CrossSim 写入误差强度下的条件模型。

方案 A：不再在 PyTorch 原始权重域手动加噪声，而是把每个 write_noise_std
映射为 CrossSim 内部的 programming_error_std。该脚本同时固定 ADC/DAC/read_noise
基线，使 CS-only、CMM、CMM-CS 的 write_noise 行成为单变量对比。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认 write_noise sweep，每个噪声强度跑 3 个 seed
#   python -m src.test_crosssim_write_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt
#
#   :: 自定义噪声列表和 seed，跳过已经存在的 checkpoint
#   python -m src.test_crosssim_write_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_write_noise_conditions ^
#       --write-noise-stds 0,1e-3,1e-2 --seeds 1,2,3 --skip-existing

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
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
    """解析 CrossSim 内部写入误差 sweep 参数。"""
    parser = argparse.ArgumentParser(
        description="通过 CrossSim programming_error 构建不同写入精度等级的 CrossSim 条件模型"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/crosssim_write_noise_conditions"),
        help="条件模型输出目录",
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
    parser.add_argument("--tile-rows", type=int, default=128, help="交叉阵列最大行数；本实验会固定为 128")
    parser.add_argument("--tile-cols", type=int, default=128, help="交叉阵列最大列数；本实验会固定为 128")
    parser.add_argument("--adc-resolution", type=int, default=0, help="ADC 分辨率；本实验会固定为理想 0")
    parser.add_argument("--dac-resolution", type=int, default=0, help="DAC 分辨率；本实验会固定为理想 0")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数")
    parser.add_argument("--rmin", type=float, default=1e3, help="器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--read-noise-std", type=float, default=1e-4, help="固定读噪声强度；本实验会固定为 1e-4")
    parser.add_argument(
        "--write-noise-stds",
        type=str,
        default="0,1e-4,3e-4,1e-3,3e-3,1e-2",
        help="逗号分隔的写入噪声标准差列表",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3", help="逗号分隔的随机种子列表")
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，覆盖 CrossSim/PyTorch 可能使用的随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(raw: str, name: str) -> List[float]:
    """解析逗号分隔的非负 float 列表。"""
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"--{name} 至少需要一个值。")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} 不能为负数。")
    return values


def parse_seed_list(raw: str) -> List[int]:
    """解析逗号分隔的 seed 列表。"""
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds 至少需要一个值。")
    return seeds


def noise_condition_name(write_noise_std: float) -> str:
    """生成稳定、可读的 write_noise 条件名。"""
    if write_noise_std == 0:
        return "write-noise-0"
    return f"write-noise-{write_noise_std:.0e}"


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """遍历写入噪声条件，构建并保存 CrossSim checkpoint。"""
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    # 固定本实验的非 sweep 变量，避免 Table XII 的 write_noise 行混入 ADC/DAC/read_noise 差异。
    args.tile_rows = 128
    args.tile_cols = 128
    args.cell_bits = 0
    args.adc_resolution = 0
    args.dac_resolution = 0
    args.read_noise_std = 1e-4

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_noise_stds = parse_float_list(args.write_noise_stds, "write-noise-stds")
    seeds = parse_seed_list(args.seeds)
    rows: List[Dict[str, str]] = []
    groups: Dict[str, Dict[str, object]] = {}

    for noise_std in write_noise_stds:
        # 方案 A：write_noise_std 只作为实验语义名；实际噪声由 CrossSim
        # NormalIndependentDevice 在电导域注入，避免原始权重域尺度不一致。
        args.programming_error_std = noise_std
        base_condition = noise_condition_name(noise_std)
        groups[base_condition] = {
            "condition": base_condition,
            "write_noise_std": str(noise_std),
            "crosssim_programming_error_std": str(noise_std),
            "seeds": [str(seed) for seed in seeds],
            "checkpoints": [],
        }

        for seed in seeds:
            set_seed(seed)
            condition = f"{base_condition}_seed-{seed}"
            out_path = args.output_dir / f"caption_transformer_{condition}_write_noise_crosssim.pt"

            if args.skip_existing and out_path.exists():
                print(f"[skip] {condition} -> already exists: {out_path}")
                rows.append(
                    {
                        "condition": condition,
                        "write_noise_group": base_condition,
                        "write_noise_std": str(noise_std),
                        "crosssim_programming_error_std": str(noise_std),
                        "seed": str(seed),
                        "checkpoint": str(out_path),
                    }
                )
                groups[base_condition]["checkpoints"].append(str(out_path))
                continue

            print(f"[build] {condition} write_noise_std={noise_std}, programming_error_std={noise_std}, seed={seed}")
            t0 = time.time()

            crosssim_model = build_crosssim_model(
                model=base_model,
                scope=args.scope,
                tile_shape=(args.tile_rows, args.tile_cols),
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
                    "write_noise_group": base_condition,
                    "write_noise_std": noise_std,
                    "crosssim_programming_error_std": noise_std,
                    # 兼容现有指标汇总脚本读取 noise_std 的习惯。
                    "noise_std": noise_std,
                    "seed": seed,
                    "cell_bits": args.cell_bits,
                    "adc_resolution": args.adc_resolution,
                    "dac_resolution": args.dac_resolution,
                    "read_noise_std": args.read_noise_std,
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
                    "write_noise_group": base_condition,
                    "write_noise_std": str(noise_std),
                    "crosssim_programming_error_std": str(noise_std),
                    "seed": str(seed),
                    "tile_shape": "128x128",
                    "adc_resolution": "0",
                    "dac_resolution": "0",
                    "cell_bits": "0",
                    "read_noise_std": "0.0001",
                    "checkpoint": str(out_path),
                }
            )
            groups[base_condition]["checkpoints"].append(str(out_path))
            print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

            del crosssim_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest_path = args.output_dir / "conditions_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    groups_path = args.output_dir / "replicate_groups_manifest.json"
    with groups_path.open("w", encoding="utf-8") as f:
        json.dump(list(groups.values()), f, ensure_ascii=False, indent=2)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved replicate groups: {groups_path}")
    return rows


def main() -> None:
    """主入口：构建写入噪声条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CrossSim write-noise condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: write_noise_std={row['write_noise_std']}, "
            f"seed={row['seed']} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
