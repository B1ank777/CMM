"""
批量构建不同 DAC 分辨率下的 CrossSim 条件模型。

默认只改变 DAC，ADC 固定在中等非理想基线（10 bit）。
如需隔离纯 DAC，请显式传入 --adc-resolution 0。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认 DAC 分辨率扫描（12, 10, 8, 6, 4 bit），含理想基线
#   python -m src.test_crosssim_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_dac_conditions --save-baseline-crosssim
#
#   :: 自定义 DAC 分辨率集合，跳过已存在的文件
#   python -m src.test_crosssim_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_dac_conditions ^
#       --resolutions 8,6,4 --skip-existing
#
#   :: GPU 加速，配合固定 ADC=8 bit 做 ADC/DAC 交叉消融
#   python -m src.test_crosssim_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_dac_adc8 ^
#       --device cuda --use-gpu --adc-resolution 8 --resolutions 8,6,4
#
#   :: 小 tile + 单元量化，仅映射 decoder 层（不含 LM head）
#   python -m src.test_crosssim_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_dac_layers ^
#       --scope layers_only --tile-rows 64 --tile-cols 64 --cell-bits 4

from __future__ import annotations

import argparse
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
    """解析 DAC 分辨率扫描参数。"""
    parser = argparse.ArgumentParser(description="Build CrossSim checkpoints for DAC resolution sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/crosssim_dac_conditions"),
        help="DAC 条件模型输出目录",
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
    parser.add_argument("--adc-resolution", type=int, default=10, help="固定 ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数")
    parser.add_argument("--rmin", type=float, default=1e3, help="器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="读噪声强度；0 表示关闭")
    parser.add_argument("--programming-error-std", type=float, default=0.0, help="CrossSim 内部写入误差；0 表示关闭")
    parser.add_argument(
        "--resolutions",
        type=str,
        default="12,10,8,6,4",
        help="逗号分隔的 DAC bit 列表，例如 12,10,8,6,4",
    )
    parser.add_argument(
        "--save-baseline-crosssim",
        action="store_true",
        help="同时保存 DAC=0、ADC=固定值的 CrossSim 基准模型",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def parse_resolutions(raw: str) -> List[int]:
    """将命令行 bit 列表解析为整数，并检查范围。"""
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--resolutions 至少需要一个 DAC bit 值。")
    if any(value < 0 for value in values):
        raise ValueError("DAC bit 值不能为负数。")
    return values


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 DAC 分辨率逐个构建 CrossSim checkpoint。"""
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    # 加载数字基线模型并冻结编码器
    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 组装条件列表：可选基线 + 扫描的 DAC 分辨率
    conditions: List[tuple[str, int]] = []
    if args.save_baseline_crosssim:
        conditions.append(("baseline-crosssim", 0))
    conditions.extend((f"dac-{bits}", bits) for bits in parse_resolutions(args.resolutions))

    rows: List[Dict[str, str]] = []
    for condition, dac_bits in conditions:
        out_path = args.output_dir / f"caption_transformer_{condition}_crosssim.pt"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "adc_resolution": str(args.adc_resolution),
                    "dac_resolution": str(dac_bits),
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} adc_resolution={args.adc_resolution}, dac_resolution={dac_bits}")
        t0 = time.time()

        # 以当前 DAC 分辨率进行 CrossSim 映射（ADC 保持固定值）
        args.dac_resolution = dac_bits
        crosssim_model = build_crosssim_model(
            model=base_model,
            scope=args.scope,
            tile_shape=(args.tile_rows, args.tile_cols),
            adc_resolution=args.adc_resolution,
            dac_resolution=dac_bits,
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

        # 保存 checkpoint，记录 ADC/DAC 条件元数据
        torch.save(
            {
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "adc_resolution": args.adc_resolution,
                "dac_resolution": dac_bits,
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
                "adc_resolution": str(args.adc_resolution),
                "dac_resolution": str(dac_bits),
                "checkpoint": str(out_path),
            }
        )
        print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

        # 释放当前模型，清理 GPU 缓存
        del crosssim_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 输出 manifest，记录所有构建条件的索引
    manifest_path = args.output_dir / "conditions_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")
    return rows


def main() -> None:
    """主入口：构建 DAC 分辨率条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CrossSim DAC condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: adc={row['adc_resolution']}, "
            f"dac={row['dac_resolution']} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
