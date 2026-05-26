"""
构建理想 CMM-on-CrossSim 基线 checkpoint。

该脚本只生成第一阶段正式基线：decoder_only、128x128、ADC/DAC=0/0、
cell_bits=0、无读写噪声。后续 cell_bits/ADC/DAC/noise/array sweep 可以复用
map_cmm_crosssim.py 的 checkpoint 格式继续扩展。
"""

# 运行示例（Windows，使用 ^ 续行）：
#   :: 生成理想 CMM-on-CrossSim 基线
#   python -m src.test_cmm_crosssim_baseline --checkpoint checkpoints\caption_transformer_epoch_10.pt
#
#   :: 指定输出目录，若已存在则跳过
#   python -m src.test_cmm_crosssim_baseline --checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --output-dir checkpoints\cmm_crosssim_conditions --skip-existing

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

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
    """解析 CMM-on-CrossSim 理想基线构建参数。"""
    parser = argparse.ArgumentParser(description="Build ideal CMM-on-CrossSim baseline checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cmm_crosssim_conditions"),
        help="CMM-on-CrossSim 条件模型输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="映射与推理设备；cuda 会启用 CrossSim GPU 后端",
    )
    parser.add_argument("--use-gpu", action="store_true", help="显式要求 CrossSim 使用 GPU")
    parser.add_argument("--skip-existing", action="store_true", help="若理想基线 checkpoint 已存在则跳过")
    return parser.parse_args()


def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """构建理想 CMM-on-CrossSim checkpoint 并写出 manifest。"""
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition = "cmm-crosssim-ideal"
    out_path = args.output_dir / "caption_transformer_cmm_crosssim_ideal.pt"

    rows: List[Dict[str, str]] = []
    if args.skip_existing and out_path.exists():
        print(f"[skip] {condition} -> already exists: {out_path}")
        rows.append({"condition": condition, "checkpoint": str(out_path)})
    else:
        payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
        base_model = build_model_from_payload(payload)
        base_model.load_state_dict(state_dict)
        freeze_encoder(base_model)
        base_model.to(device)
        base_model.eval()

        # 固定第一阶段理想基线参数，避免和后续 sweep 条件混在一起。
        cmm_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            output=out_path,
            device=args.device,
            use_gpu=args.use_gpu,
            scope="decoder_only",
            tile_rows=128,
            tile_cols=128,
            rmin=1e3,
            rmax=1e5,
            cell_bits=0,
            adc_resolution=0,
            dac_resolution=0,
            read_noise_std=0.0,
            write_noise_std=0.0,
            programming_error_std=0.0,
            bias_rows=0,
        )

        print(f"[build] {condition}")
        t0 = time.time()
        num_mapped_linear = count_linear_layers_by_scope(base_model, cmm_args.scope)
        crosssim_model = build_cmm_crosssim_model(base_model, cmm_args, use_gpu)
        crosssim_model.to(device)
        crosssim_model.eval()

        torch.save(
            {
                "format": "cmm_crosssim_v1",
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "model_config": payload.get("model_config"),
                "vocab_stoi": payload.get("vocab_stoi"),
                "crosssim_model_state_dict": crosssim_model.state_dict(),
                "crosssim_args": make_crosssim_args(cmm_args, use_gpu),
                "cmm_crosssim_args": make_cmm_crosssim_args(cmm_args, use_gpu),
                "num_mapped_linear": num_mapped_linear,
            },
            out_path,
        )

        rows.append(
            {
                "condition": condition,
                "scope": cmm_args.scope,
                "tile_shape": "128x128",
                "adc_resolution": "0",
                "dac_resolution": "0",
                "cell_bits": "0",
                "write_noise_std": "0.0",
                "read_noise_std": "0.0",
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
    """主入口：构建并汇报理想 CMM-on-CrossSim 基线。"""
    rows = build_condition_models(parse_args())
    print("=== Built CMM-on-CrossSim baseline models ===")
    for row in rows:
        print(
            f"{row['condition']}: num_mapped_linear={row.get('num_mapped_linear', 'unknown')} "
            f"-> {row['checkpoint']}"
        )


if __name__ == "__main__":
    main()
