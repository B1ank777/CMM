"""
批量构建不同手动写入噪声强度下的 CrossSim 条件模型。

该脚本先在 PyTorch 权重上注入高斯噪声，再映射为 CrossSim AnalogLinear。
它用于模拟外部编程误差，不等同于 CrossSim 内部的 programming_error 模型。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List

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
    """解析手动写入噪声 sweep 参数。"""
    parser = argparse.ArgumentParser(
        description="通过手动权重噪声构建不同写入精度等级的 CrossSim 条件模型"
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
    parser.add_argument("--tile-rows", type=int, default=128, help="交叉阵列最大行数")
    parser.add_argument("--tile-cols", type=int, default=128, help="交叉阵列最大列数")
    parser.add_argument("--adc-resolution", type=int, default=10, help="ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=12, help="DAC 分辨率；0 表示理想 DAC")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数")
    parser.add_argument("--rmin", type=float, default=1e3, help="器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="固定读噪声强度；0 表示关闭")
    parser.add_argument("--programming-error-std", type=float, default=0.0, help="CrossSim 内部写入误差；0 表示关闭")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证噪声注入可复现")
    parser.add_argument(
        "--save-baseline-crosssim",
        action="store_true",
        help="同时保存 noise_std=0 的 CrossSim 基准模型",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="",
        help="逗号分隔的子集，例如 it-10,it-9,it-8；为空则运行全部默认条件",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def inject_write_noise(model: torch.nn.Module, noise_std: float) -> torch.nn.Module:
    """向所有矩阵权重注入高斯噪声，模拟手动写入误差。"""
    noisy = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in noisy.named_parameters():
            if "weight" in name and param.dim() >= 2:
                param.add_(torch.randn_like(param) * noise_std)
    return noisy


def default_noise_settings() -> Dict[str, float]:
    """返回默认写入精度等级到噪声标准差的映射。"""
    return {
        "it-10": 1e-6,
        "it-9": 1e-5,
        "it-8": 1e-4,
        "it-7": 1e-3,
        "it-6": 1e-2,
    }


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """遍历写入噪声条件，构建并保存 CrossSim checkpoint。"""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = default_noise_settings()
    rows: List[Dict[str, str]] = []
    if args.save_baseline_crosssim:
        conditions = {"baseline-crosssim": 0.0, **conditions}

    if args.conditions.strip():
        selected = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
        unknown = [condition for condition in selected if condition not in conditions]
        if unknown:
            raise ValueError(f"Unknown conditions: {unknown}. Available: {list(conditions.keys())}")
        conditions = {key: conditions[key] for key in selected}

    for condition, noise_std in conditions.items():
        out_path = args.output_dir / f"caption_transformer_{condition}_write_noise_crosssim.pt"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append({"condition": condition, "write_noise_std": str(noise_std), "checkpoint": str(out_path)})
            continue

        print(f"[build] {condition} write_noise_std={noise_std}")
        t0 = time.time()

        noisy_model = inject_write_noise(base_model, noise_std=noise_std)
        noisy_model.to(device)

        crosssim_model = build_crosssim_model(
            model=noisy_model,
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
                "write_noise_std": noise_std,
                # 兼容现有指标汇总脚本读取 noise_std 的习惯。
                "noise_std": noise_std,
                "model_config": payload.get("model_config"),
                "vocab_stoi": payload.get("vocab_stoi"),
                "crosssim_model_state_dict": crosssim_model.state_dict(),
                "crosssim_args": make_crosssim_args(args, use_gpu),
            },
            out_path,
        )

        rows.append({"condition": condition, "write_noise_std": str(noise_std), "checkpoint": str(out_path)})
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
    """主入口：构建写入噪声条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CrossSim write-noise condition models ===")
    for row in rows:
        print(f"{row['condition']}: write_noise_std={row['write_noise_std']} -> {row['checkpoint']}")


if __name__ == "__main__":
    main()
