"""
批量构建不同 CMM write_noise_std 下的条件模型。

固定 ADC/DAC=0/0、cell_bits = 0、read_noise_std = 1e-4，只扫描写入噪声。写入噪声在映射时
固定进入 r_pos/r_neg，因此每个噪声强度默认构建 seed = 1, 2, 3 三个 checkpoint。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 默认 write_noise sweep，每个噪声强度跑 3 个 seed
#   python -m src.test_cmm_write_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_write_noise_conditions
#
#   :: 自定义噪声列表和 seed，跳过已存在的 checkpoint
#   python -m src.test_cmm_write_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_write_noise_conditions ^
#       --write-noise-stds 0,1e-3,1e-2 --seeds 1,2,3 --skip-existing
#
#   :: 小 tile，仅映射 decoder layers
#   python -m src.test_cmm_write_noise_conditions --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_write_noise_layers ^
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
    """解析 CMM write_noise sweep 参数。"""
    parser = argparse.ArgumentParser(description="Build CMM checkpoints for write_noise sweep")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_write_noise_conditions"),
        help="CMM write_noise 条件模型输出目录",
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
        "--write-noise-stds",
        type=str,
        default="0,1e-4,3e-4,1e-3,3e-3,1e-2",
        help="逗号分隔的写入噪声标准差列表",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3", help="逗号分隔的随机种子列表")
    parser.add_argument("--skip-existing", action="store_true", help="若目标 checkpoint 已存在则跳过该条件")
    return parser.parse_args()


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


def set_seed(seed: int) -> None:
    """设置 PyTorch 随机种子，保证 r_pos/r_neg 写入噪声可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """按 write_noise_std 和 seed 逐个构建 CMM checkpoint。"""
    device = torch.device(args.device)

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

    for write_noise_std in write_noise_stds:
        args.write_noise_std = write_noise_std
        base_condition = noise_condition_name(write_noise_std)
        groups[base_condition] = {
            "condition": base_condition,
            "write_noise_std": str(write_noise_std),
            "seeds": [str(seed) for seed in seeds],
            "checkpoints": [],
        }

        for seed in seeds:
            set_seed(seed)
            condition = f"{base_condition}_seed-{seed}"
            out_path = args.output_dir / f"caption_transformer_{condition}_cmm.pt"

            if args.skip_existing and out_path.exists():
                print(f"[skip] {condition} -> already exists: {out_path}")
                rows.append(
                    {
                        "condition": condition,
                        "write_noise_group": base_condition,
                        "write_noise_std": str(write_noise_std),
                        "seed": str(seed),
                        "tile_shape": "128x128",
                        "cell_bits": "0",
                        "adc_resolution": "0",
                        "dac_resolution": "0",
                        "read_noise_std": "0.0001",
                        "checkpoint": str(out_path),
                    }
                )
                groups[base_condition]["checkpoints"].append(str(out_path))
                continue

            print(f"[build] {condition} write_noise_std={write_noise_std}, seed={seed}")
            t0 = time.time()

            # 写入噪声在 build_cmm_model 内部注入到 r_pos/r_neg，并随 checkpoint 固化。
            cmm_model = build_cmm_model(
                model=base_model,
                scope=args.scope,
                tile_shape=(args.tile_rows, args.tile_cols),
                rmin=args.rmin,
                rmax=args.rmax,
                cell_bits=0,
                write_noise_std=write_noise_std,
                read_noise_std=args.read_noise_std,
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
                    "write_noise_group": base_condition,
                    "write_noise_std": write_noise_std,
                    "seed": seed,
                    "cell_bits": 0,
                    "adc_resolution": args.adc_resolution,
                    "dac_resolution": args.dac_resolution,
                    "read_noise_std": args.read_noise_std,
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
                    "write_noise_group": base_condition,
                    "write_noise_std": str(write_noise_std),
                    "seed": str(seed),
                    "tile_shape": "128x128",
                    "cell_bits": "0",
                    "adc_resolution": "0",
                    "dac_resolution": "0",
                    "read_noise_std": "0.0001",
                    "num_cmm_linear": str(num_cmm_linear),
                    "checkpoint": str(out_path),
                }
            )
            groups[base_condition]["checkpoints"].append(str(out_path))
            print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

            del cmm_model
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
    """主入口：构建 CMM write_noise 条件模型并输出 manifest。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM write_noise condition models ===")
    for row in rows:
        print(
            f"{row['condition']}: write_noise_std={row['write_noise_std']}, "
            f"seed={row['seed']} -> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
