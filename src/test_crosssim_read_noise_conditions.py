"""
批量构建不同 CrossSim read_noise_std 下的条件模型。

该脚本不改动 PyTorch 权重，只通过 CrossSim 的 read_noise 模型改变读噪声强度。
默认 ADC/DAC 使用中等非理想基线（ADC=10 bit, DAC=12 bit）。
如需隔离纯读噪声，请显式传入 --adc-resolution 0 --dac-resolution 0。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认读噪声扫描（0, 1e-5, 1e-4, 1e-3, 1e-2）
#   python -m src.test_crosssim_read_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_read_noise_conditions
#
#   :: 自定义读噪声列表，跳过已存在的文件
#   python -m src.test_crosssim_read_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_read_noise_conditions ^
#       --read-noise-stds 0,5e-6,1e-5,5e-5,1e-4 --skip-existing
#
#   :: GPU 加速，配合 ADC/DAC 量化做噪声 + 量化交叉消融
#   python -m src.test_crosssim_read_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_rn_adc8_dac4 ^
#       --device cuda --use-gpu --adc-resolution 8 --dac-resolution 4 ^
#       --read-noise-stds 0,1e-5,1e-4,1e-3
#
#   :: 小 tile + 器件量化，仅映射 decoder 层
#   python -m src.test_crosssim_read_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\crosssim_rn_layers ^
#       --scope layers_only --tile-rows 64 --tile-cols 64 --cell-bits 8

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
    """解析读噪声 sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CrossSim checkpoints for read-noise sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/crosssim_read_noise_conditions"),
        help="读噪声条件模型输出目录",
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
    parser.add_argument("--programming-error-std", type=float, default=0.0, help="CrossSim 内部写入误差；0 表示关闭")
    parser.add_argument(
        "--read-noise-stds",
        type=str,
        default="0,1e-5,1e-4,1e-3,1e-2",
        help="逗号分隔的读噪声标准差列表",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def parse_noise_stds(raw: str) -> List[float]:
    """将命令行读噪声列表解析为 float。"""
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--read-noise-stds 至少需要一个值。")
    if any(value < 0 for value in values):
        raise ValueError("read_noise_std 不能为负数。")
    return values


def condition_name(read_noise_std: float) -> str:
    """生成稳定、可读的读噪声条件名。

    0 值使用 "read-noise-0"，非零值使用科学记数法（如 read-noise-1e-04），
    确保文件名不含小数点或多余零，便于后续脚本解析。
    """
    if read_noise_std == 0:
        return "read-noise-0"
    return f"read-noise-{read_noise_std:.0e}"


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 read_noise_std 逐个构建 CrossSim checkpoint。"""
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

    rows: List[Dict[str, str]] = []
    for read_noise_std in parse_noise_stds(args.read_noise_stds):
        # make_crosssim_args 需要当前单个 read_noise_std，而不是 sweep 列表。
        args.read_noise_std = read_noise_std
        condition = condition_name(read_noise_std)
        out_path = args.output_dir / f"caption_transformer_{condition}_crosssim.pt"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "read_noise_std": str(read_noise_std),
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} read_noise_std={read_noise_std}")
        t0 = time.time()

        # 以当前 read_noise_std 进行 CrossSim 映射（不改动原始权重）
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
            read_noise_std=read_noise_std,
            programming_error_std=args.programming_error_std,
        )
        crosssim_model.to(device)
        crosssim_model.eval()

        # 保存 checkpoint，记录读噪声条件元数据
        torch.save(
            {
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "read_noise_std": read_noise_std,
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
                "read_noise_std": str(read_noise_std),
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
    """主入口：构建读噪声条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CrossSim read-noise condition models ===")
    for row in rows:
        print(f"{row['condition']}: read_noise_std={row['read_noise_std']} -> {row['checkpoint']}")


if __name__ == "__main__":
    main()
