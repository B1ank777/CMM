"""
批量构建不同 CMM tile 尺寸下的条件模型。

默认只改变 tile_rows/tile_cols，ADC 固定为 8 bit，DAC 固定为理想 0 bit。
这样 tile 切分会通过 partial-sum ADC 量化真实影响输出，而不是只作为元数据。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认方阵 tile 扫描（64, 128, 256, 512），ADC=8 bit
#   python -m src.test_cmm_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_array_size_conditions
#
#   :: 自定义方阵尺寸，跳过已存在的 checkpoint
#   python -m src.test_cmm_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_array_size_conditions ^
#       --array-sizes 32,64,128 --skip-existing
#
#   :: 矩形 tile 尺寸
#   python -m src.test_cmm_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_array_rect ^
#       --rect-array-sizes 64x128,128x256,256x512
#
#   :: 理想 ADC，仅验证分块结构本身是否保持等价
#   python -m src.test_cmm_array_size_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_array_ideal --adc-resolution 0

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .cmm import count_cmm_linear_layers
from .map_cmm import build_cmm_model, make_cmm_args
from .map_crosssim import build_model_from_payload, freeze_encoder, load_checkpoint


def parse_args() -> argparse.Namespace:
    """解析 CMM tile size sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CMM checkpoints for tile-size sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_array_size_conditions"),
        help="CMM tile size 条件模型输出目录",
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
    parser.add_argument("--rmin", type=float, default=1e3, help="Ron，器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="Roff，器件最大电阻")
    parser.add_argument("--adc-resolution", type=int, default=8, help="固定 ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=0, help="固定 DAC 分辨率；0 表示理想 DAC")
    parser.add_argument(
        "--array-sizes",
        type=str,
        default="64,128,256,512",
        help="逗号分隔的方阵 tile 尺寸列表，例如 64,128,256,512",
    )
    parser.add_argument(
        "--rect-array-sizes",
        type=str,
        default="",
        help="可选矩形 tile 尺寸列表，例如 64x128,128x256；若提供则覆盖 --array-sizes",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


def parse_array_sizes(args: argparse.Namespace) -> List[Tuple[int, int]]:
    """解析方阵或矩形 tile 尺寸列表。"""
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
        raise ValueError("至少需要一个 tile size。")
    if any(rows <= 0 or cols <= 0 for rows, cols in sizes):
        raise ValueError("tile size 的行列数必须为正整数。")
    return sizes


def condition_name(tile_rows: int, tile_cols: int) -> str:
    """生成稳定的 tile size 条件名。"""
    return f"tile-{tile_rows}x{tile_cols}"


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 tile size 逐个构建 CMM checkpoint。"""
    device = torch.device(args.device)

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for tile_rows, tile_cols in parse_array_sizes(args):
        # make_cmm_args 需要当前单个 tile_rows/tile_cols。
        args.tile_rows = tile_rows
        args.tile_cols = tile_cols
        args.cell_bits = 0
        args.write_noise_std = 0.0
        args.read_noise_std = 0.0

        condition = condition_name(tile_rows, tile_cols)
        out_path = args.output_dir / f"caption_transformer_{condition}_cmm.pt"

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
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(
            f"[build] {condition} tile_shape=({tile_rows}, {tile_cols}), "
            f"adc={args.adc_resolution}, dac={args.dac_resolution}"
        )
        t0 = time.time()

        cmm_model = build_cmm_model(
            model=base_model,
            scope=args.scope,
            tile_shape=(tile_rows, tile_cols),
            rmin=args.rmin,
            rmax=args.rmax,
            cell_bits=0,
            write_noise_std=0.0,
            read_noise_std=0.0,
            adc_resolution=args.adc_resolution,
            dac_resolution=args.dac_resolution,
        )
        cmm_model.to(device)
        cmm_model.eval()
        num_cmm_linear = count_cmm_linear_layers(cmm_model)

        torch.save(
            {
                "format": "cmm_v1",
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
                "array_size": f"{tile_rows}x{tile_cols}",
                "adc_resolution": args.adc_resolution,
                "dac_resolution": args.dac_resolution,
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
                "tile_rows": str(tile_rows),
                "tile_cols": str(tile_cols),
                "array_size": f"{tile_rows}x{tile_cols}",
                "adc_resolution": str(args.adc_resolution),
                "dac_resolution": str(args.dac_resolution),
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
    """主入口：构建 CMM tile size 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM tile-size condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: array_size={row['array_size']}, "
            f"adc={row['adc_resolution']}, dac={row['dac_resolution']} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
