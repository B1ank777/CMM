"""
批量构建不同 CMM cell_bits 下的条件模型。

该脚本使用论文式 CMM 映射，不依赖 CrossSim simulator。默认扫描
cell_bits = 0, 2, 3, 4, 6, 8，其中 0 表示连续忆阻器状态。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认 cell_bits 扫描（0, 2, 3, 4, 6, 8）
#   python -m src.test_cmm_cell_bits_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_cell_bits_conditions
#
#   :: 自定义 bit 列表，跳过已存在的 checkpoint
#   python -m src.test_cmm_cell_bits_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_cell_bits_conditions ^
#       --cell-bits-list 0,4,8 --skip-existing
#
#   :: 小 tile + 固定读/写噪声，仅映射 decoder layers
#   python -m src.test_cmm_cell_bits_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_cell_bits_layers ^
#       --scope layers_only --tile-rows 64 --tile-cols 64 ^
#       --write-noise-std 0.001 --read-noise-std 0.0001

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
    """解析 CMM cell_bits sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CMM checkpoints for cell_bits sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_cell_bits_conditions"),
        help="CMM cell_bits 条件模型输出目录",
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
    parser.add_argument("--write-noise-std", type=float, default=0.0, help="写入噪声强度，作用在内部状态 r 上")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="读噪声强度，作用在推理时 Rmem 上")
    parser.add_argument(
        "--cell-bits-list",
        type=str,
        default="0,2,3,4,6,8",
        help="逗号分隔的 CMM cell_bits 列表；0 表示连续状态",
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


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 cell_bits 逐个构建 CMM checkpoint。"""
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
    for cell_bits in parse_cell_bits(args.cell_bits_list):
        # make_cmm_args 需要当前单个 cell_bits，而不是 sweep 列表。
        args.cell_bits = cell_bits
        condition = condition_name(cell_bits)
        out_path = args.output_dir / f"caption_transformer_{condition}_cmm.pt"

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

        # 以当前 cell_bits 进行 CMM 映射。0 表示连续状态，不做忆阻器状态量化。
        cmm_model = build_cmm_model(
            model=base_model,
            scope=args.scope,
            tile_shape=(args.tile_rows, args.tile_cols),
            rmin=args.rmin,
            rmax=args.rmax,
            cell_bits=cell_bits,
            write_noise_std=args.write_noise_std,
            read_noise_std=args.read_noise_std,
        )
        cmm_model.to(device)
        cmm_model.eval()
        num_cmm_linear = count_cmm_linear_layers(cmm_model)

        torch.save(
            {
                "format": "cmm_v1",
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "cell_bits": cell_bits,
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
                "cell_bits": str(cell_bits),
                "num_cmm_linear": str(num_cmm_linear),
                "checkpoint": str(out_path),
            }
        )
        print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

        # 释放当前模型，清理 GPU 缓存。
        del cmm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = args.output_dir / "conditions_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")
    return rows


def main() -> None:
    """主入口：构建 CMM cell_bits 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM cell_bits condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: cell_bits={row['cell_bits']}, "
            f"num_cmm_linear={row.get('num_cmm_linear', 'unknown')} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
