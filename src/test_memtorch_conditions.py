"""
批量构建不同写噪声强度下的忆阻器条件模型。

功能:
    对原始模型权重注入不同程度的高斯噪声（模拟忆阻器编程误差），
    然后分别映射为忆阻器交叉阵列模型，保存用于鲁棒性评估。

动机:
    真实的忆阻器编程过程中，目标电导值与实际写入值之间存在误差。
    不同噪声标准差对应不同的编程精度等级（it-10 到 it-6），
    通过对比这些条件模型在验证集上的性能退化，可以：
        1. 评估模型对忆阻器编程噪声的容忍度
        2. 确定可接受的编程精度下限
        3. 为硬件设计提供精度需求参考

条件说明:
    it-10: noise_std=1e-6  （极高精度，近乎理想编程）
    it-9:  noise_std=1e-5  （很高精度）
    it-8:  noise_std=1e-4  （高精度）
    it-7:  noise_std=1e-3  （中等精度）
    it-6:  noise_std=1e-2  （较低精度，噪声较大）

运行方式:
    python -m src.test_memtorch_conditions --checkpoint checkpoints/caption_transformer_epoch_10.pt --output-dir checkpoints/memtorch_conditions --save-baseline-mem
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from .map_memtorch import (
    build_memristive_model,
    build_model_from_payload,
    freeze_encoder,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="通过注入写噪声构建不同编程精度等级的 MemTorch 条件模型"
    )

    # --- 文件路径 ---
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="原始训练检查点路径",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("checkpoints/memtorch_conditions"),
        help="条件模型输出目录",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    parser.add_argument(
        "--map-device", type=str, default="cpu",
        help="执行 memtorch patch/mapping 的设备（建议 cpu）",
    )

    # --- 忆阻器映射参数 ---
    parser.add_argument(
        "--use-bindings", action="store_true",
        help="是否使用 memtorch 的 C++ binding 加速",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="MemTorch patch scope",
    )
    parser.add_argument(
        "--tile-rows", type=int, default=128,
        help="交叉阵列分块行数",
    )
    parser.add_argument(
        "--tile-cols", type=int, default=128,
        help="交叉阵列分块列数",
    )
    parser.add_argument(
        "--max-input-voltage", type=float, default=0.3,
        help="最大输入电压 (V)",
    )
    parser.add_argument(
        "--adc-resolution", type=int, default=8,
        help="ADC 分辨率 (bit)",
    )
    parser.add_argument(
        "--ron", type=float, default=1e2,
        help="忆阻器低阻态电阻 (Ω)",
    )
    parser.add_argument(
        "--roff", type=float, default=1e4,
        help="忆阻器高阻态电阻 (Ω)",
    )

    # --- 噪声与输出控制 ---
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，保证噪声注入可复现",
    )
    parser.add_argument(
        "--save-baseline-mem", action="store_true",
        help="同时保存 noise_std=0 的基准忆阻器模型（无噪声映射）",
    )
    parser.add_argument(
        "--conditions", type=str, default="",
        help="逗号分隔的子集，例如 it-10,it-9,it-8；为空则运行全部默认条件",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="若目标 checkpoint 已存在则跳过该条件",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 写噪声注入
# ---------------------------------------------------------------------------

def inject_write_noise(model: torch.nn.Module, noise_std: float) -> torch.nn.Module:
    """向模型的所有权重参数注入高斯噪声，模拟忆阻器编程误差。

    仅对 dim >= 2 的 weight 参数注入噪声（线性层权重矩阵），
    跳过 bias 和一维参数（如 LayerNorm 的 weight）。
    使用深拷贝避免污染原始模型。

    参数:
        model: 原始模型。
        noise_std: 高斯噪声的标准差，越大表示编程精度越差。

    返回:
        权重被噪声扰动后的模型副本。
    """
    noisy = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in noisy.named_parameters():
            if "weight" in name and param.dim() >= 2:
                param.add_(torch.randn_like(param) * noise_std)
    return noisy


# ---------------------------------------------------------------------------
# 噪声等级定义
# ---------------------------------------------------------------------------

def default_noise_settings() -> Dict[str, float]:
    """返回默认的编程精度等级 → 噪声标准差的映射。

    it-10 到 it-6 分别对应从极高精度到较低精度的编程条件，
    噪声量级从 1e-6 指数递增到 1e-2。
    """
    return {
        "it-10": 1e-6,  # 极高精度
        "it-9": 1e-5,   # 很高精度
        "it-8": 1e-4,   # 高精度
        "it-7": 1e-3,   # 中等精度
        "it-6": 1e-2,   # 较低精度
    }


# ---------------------------------------------------------------------------
# 批量构建条件模型
# ---------------------------------------------------------------------------

def build_condition_models(args: argparse.Namespace) -> List[Dict[str, str]]:
    """遍历所有噪声等级，构建并保存对应的忆阻器条件模型。

    流程:
        1. 加载原始训练模型
        2. 对每种噪声等级：注入噪声 → 映射忆阻器 → 保存检查点
        3. 生成 conditions_manifest.json 索引文件

    返回:
        条件模型清单，每项包含 condition、noise_std、checkpoint 路径。
    """
    # 固定随机种子，保证噪声注入可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载原始训练模型
    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(torch.device(args.map_device))
    base_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = default_noise_settings()
    rows: List[Dict[str, str]] = []

    # 可选：额外保存无噪声的基准忆阻器模型
    if args.save_baseline_mem:
        conditions = {"baseline-mem": 0.0, **conditions}

    if args.conditions.strip():
        selected = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in selected if c not in conditions]
        if unknown:
            raise ValueError(f"Unknown conditions: {unknown}. Available: {list(conditions.keys())}")
        conditions = {k: conditions[k] for k in selected}

    for condition, noise_std in conditions.items():
        out_path = args.output_dir / f"caption_transformer_{condition}_memtorch.pt"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {condition} -> already exists: {out_path}")
            rows.append(
                {
                    "condition": condition,
                    "noise_std": str(noise_std),
                    "checkpoint": str(out_path),
                }
            )
            continue

        print(f"[build] {condition} noise_std={noise_std}")
        t0 = time.time()

        # 1. 注入写噪声
        noisy_model = inject_write_noise(base_model, noise_std=noise_std)
        noisy_model.to(torch.device(args.map_device))

        # 2. 映射为忆阻器交叉阵列模型
        mem_model = build_memristive_model(
            model=noisy_model,
            use_bindings=args.use_bindings,
            scope=args.scope,
            tile_shape=(args.tile_rows, args.tile_cols),
            max_input_voltage=args.max_input_voltage,
            adc_resolution=args.adc_resolution,
            ron=args.ron,
            roff=args.roff,
        )
        mem_model.to(torch.device(args.device))
        mem_model.eval()

        # 3. 保存条件模型检查点
        torch.save(
            {
                "original_checkpoint": str(args.checkpoint),
                "condition": condition,
                "noise_std": noise_std,
                "model_config": payload.get("model_config"),
                "vocab_stoi": payload.get("vocab_stoi"),
                "mem_model_state_dict": mem_model.state_dict(),
                "memtorch_args": {
                    "use_bindings": args.use_bindings,
                    "tile_shape": (args.tile_rows, args.tile_cols),
                    "max_input_voltage": args.max_input_voltage,
                    "adc_resolution": args.adc_resolution,
                    "r_on": args.ron,
                    "r_off": args.roff,
                    "mapping_scope": args.scope,
                },
            },
            out_path,
        )

        rows.append(
            {
                "condition": condition,
                "noise_std": str(noise_std),
                "checkpoint": str(out_path),
            }
        )
        print(f"[done] {condition} in {time.time() - t0:.1f}s -> {out_path}")

        # 释放显存
        del mem_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. 保存条件清单，供后续评估脚本索引
    manifest_path = args.output_dir / "conditions_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved manifest: {manifest_path}")
    return rows


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    """主入口：解析参数 → 构建所有条件模型 → 输出清单。"""
    args = parse_args()
    rows = build_condition_models(args)
    print("=== Built condition models ===")
    for r in rows:
        print(f"{r['condition']}: noise_std={r['noise_std']} -> {r['checkpoint']}")


if __name__ == "__main__":
    main()
