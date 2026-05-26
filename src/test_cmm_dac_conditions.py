"""
批量构建不同 CMM DAC 分辨率下的条件模型。

默认只改变 DAC，ADC 固定为理想 0 bit；cell_bits/write_noise/read_noise 均固定为 0，
用于隔离观察输入 tile DAC 量化对模型的影响。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认 DAC 分辨率扫描（0, 12, 10, 8, 6, 4 bit）
#   python -m src.test_cmm_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_dac_conditions
#
#   :: 自定义 DAC 分辨率集合，跳过已存在的 checkpoint
#   python -m src.test_cmm_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_dac_conditions ^
#       --dac-resolutions 0,10,8,6,4 --skip-existing
#
#   :: 小 tile，仅映射 decoder layers
#   python -m src.test_cmm_dac_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_dac_layers ^
#       --scope layers_only --tile-rows 64 --tile-cols 64

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from .cmm import count_cmm_linear_layers
from .map_cmm import build_cmm_model, make_cmm_args
from .map_crosssim import build_model_from_payload, freeze_encoder, load_checkpoint


def parse_args() -> argparse.Namespace:
    """解析 CMM DAC sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CMM checkpoints for DAC resolution sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_dac_conditions"),
        help="CMM DAC 条件模型输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="映射设备",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="CMM 映射范围",
    )
    parser.add_argument("--tile-rows", type=int, default=128, help="CMM 阵列最大行数")
    parser.add_argument("--tile-cols", type=int, default=128, help="CMM 阵列最大列数")
    parser.add_argument("--rmin", type=float, default=1e3, help="Ron，器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="Roff，器件最大电阻")
    parser.add_argument(
        "--dac-resolutions",
        type=str,
        default="0,12,10,8,6,4",
        help="逗号分隔的 DAC bit 列表；0 表示理想 DAC",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def parse_resolutions(raw: str) -> List[int]:
    """将命令行 bit 列表解析为整数，并检查范围。"""
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--dac-resolutions 至少需要一个值。")
    if any(value < 0 for value in values):
        raise ValueError("DAC bit 值不能为负数。")
    return values


def condition_name(dac_bits: int) -> str:
    """生成稳定的 DAC 条件名。"""
    return "dac-ideal" if dac_bits == 0 else f"dac-{dac_bits}bit"


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 DAC 分辨率逐个构建 CMM checkpoint。"""
    device = torch.device(args.device)

    # 加载数字基线模型并冻结编码器
    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for dac_bits in parse_resolutions(args.dac_resolutions):
        # make_cmm_args 需要当前单个 dac_resolution，而不是 sweep 列表。
        args.cell_bits = 0
        args.write_noise_std = 0.0
        args.read_noise_std = 0.0
        args.adc_resolution = 0
        args.dac_resolution = dac_bits

        condition = condition_name(dac_bits)
        out_path = args.output_dir / f"caption_transformer_{condition}_cmm.pt"

        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "adc_resolution": "0",
                    "dac_resolution": str(dac_bits),
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} adc_resolution=0, dac_resolution={dac_bits}")
        t0 = time.time()

        cmm_model = build_cmm_model(
            model=base_model,
            scope=args.scope,
            tile_shape=(args.tile_rows, args.tile_cols),
            rmin=args.rmin,
            rmax=args.rmax,
            cell_bits=0,
            write_noise_std=0.0,
            read_noise_std=0.0,
            adc_resolution=0,
            dac_resolution=dac_bits,
        )
        cmm_model.to(device)
        cmm_model.eval()
        num_cmm_linear = count_cmm_linear_layers(cmm_model)

        torch.save(
            {
                "format": "cmm_v1",
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "adc_resolution": 0,
                "dac_resolution": dac_bits,
                "cell_bits": 0,
                "write_noise_std": 0.0,
                "read_noise_std": 0.0,
                "model_config": payload.get("model_config"),
                "vocab_stoi": payload.get("vocab_stoi"),
                "cmm_model_state_dict": cmm_model.state_dict(),
                "cmm_args": make_cmm_args(args),
                "num_cmm_linear": num_cmm_linear,
            },
            out_path,
        )

        rows.append(
            {
                "condition": condition,
                "adc_resolution": "0",
                "dac_resolution": str(dac_bits),
                "cell_bits": "0",
                "write_noise_std": "0.0",
                "read_noise_std": "0.0",
                "num_cmm_linear": str(num_cmm_linear),
                "checkpoint": str(out_path),
            }
        )
        print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

        del cmm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = args.output_dir / "conditions_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")
    return rows


def main() -> None:
    """主入口：构建 CMM DAC 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM DAC condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: adc={row['adc_resolution']}, "
            f"dac={row['dac_resolution']} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
