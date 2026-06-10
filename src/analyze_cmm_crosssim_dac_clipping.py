from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .coco_preprocess.loader import DEFAULT_COCO_ROOT
from .dac_adaptive_utils import (
    build_validation_loader_from_payload,
    layer_group,
    load_baseline_and_crosssim,
    mapped_crosssim_linears,
    quantize_symmetric,
    run_teacher_forcing_batches,
)


# 运行示例：
#   python -m src.analyze_cmm_crosssim_dac_clipping ^
#       --baseline-checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --crosssim-checkpoint checkpoints\cmm_crosssim_dac_conditions\caption_transformer_dac-12bit_cmm_crosssim.pt ^
#       --output-dir experiments\dac_range_root_cause --device cuda --max-batches 50


DEFAULT_BASELINE = Path("checkpoints/caption_transformer_epoch_10.pt")
DEFAULT_CMM_CROSSSIM_DAC12 = Path(
    "checkpoints/cmm_crosssim_dac_conditions/caption_transformer_dac-12bit_cmm_crosssim.pt"
)
DEFAULT_OUTPUT_DIR = Path("experiments/dac_range_root_cause")
DEFAULT_PLOT_LAYERS = [
    "layers.0.self_attn.q_proj",
    "layers.0.cross_attn.v_proj",
    "layers.0.ffn.0",
    "layers.0.ffn.3",
    "output_proj",
]


@dataclass
class ActivationStats:
    """单层 activation 的流式统计与少量样本缓存。"""

    layer_name: str
    count: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    sum_value: float = 0.0
    sum_square: float = 0.0
    clip_count: int = 0
    abs_values: List[np.ndarray] = field(default_factory=list)
    samples: List[np.ndarray] = field(default_factory=list)
    sample_limit: int = 200_000

    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().float().reshape(-1).cpu()
        if flat.numel() == 0:
            return
        self.count += int(flat.numel())
        self.min_value = min(self.min_value, float(flat.min()))
        self.max_value = max(self.max_value, float(flat.max()))
        self.sum_value += float(flat.sum())
        self.sum_square += float(flat.square().sum())
        self.clip_count += int(flat.abs().gt(1.0).sum())
        self.abs_values.append(flat.abs().numpy())

        # 中文注释：只缓存有限数量样本用于画直方图，避免大验证集把内存吃满。
        remain = self.sample_limit - sum(part.size for part in self.samples)
        if remain > 0:
            self.samples.append(flat[:remain].numpy())

    def row(self) -> Dict[str, float | str | int]:
        abs_all = np.concatenate(self.abs_values) if self.abs_values else np.array([], dtype=np.float32)
        mean = self.sum_value / max(self.count, 1)
        variance = self.sum_square / max(self.count, 1) - mean * mean
        return {
            "layer": self.layer_name,
            "group": layer_group(self.layer_name),
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "std": float(max(variance, 0.0) ** 0.5),
            "p99_abs": float(np.percentile(abs_all, 99)) if abs_all.size else 0.0,
            "p99.9_abs": float(np.percentile(abs_all, 99.9)) if abs_all.size else 0.0,
            "clip_rate_abs_gt_1": self.clip_count / max(self.count, 1),
            "adaptive_absmax": max(abs(self.min_value), abs(self.max_value)),
        }

    def sample_array(self) -> np.ndarray:
        return np.concatenate(self.samples) if self.samples else np.array([], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DAC clipping on CMM-CrossSim mapped Linear activations.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE, help="数字基线 checkpoint")
    parser.add_argument("--crosssim-checkpoint", type=Path, default=DEFAULT_CMM_CROSSSIM_DAC12, help="DAC-12 CMM-CrossSim checkpoint")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT, help="COCO 数据根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="统计表和分布图输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--batch-size", type=int, default=8, help="activation 采样 batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--max-batches", type=int, default=50, help="最多采样多少个 batch")
    parser.add_argument("--subset-size", type=int, default=0, help="验证集子集大小；0 表示不截断")
    parser.add_argument("--dac-bits", type=int, default=12, help="画 adaptive 量化分布时使用的 DAC bit")
    parser.add_argument("--plot-layers", type=str, default=",".join(DEFAULT_PLOT_LAYERS), help="逗号分隔的典型层名")
    parser.add_argument("--sample-limit", type=int, default=200_000, help="每层最多保留多少个值用于画图")
    parser.add_argument(
        "--activation-source",
        type=str,
        default="baseline",
        choices=["baseline", "crosssim"],
        help="activation 统计来源；baseline 表示量化前原始分布，crosssim 表示当前 DAC 模型内部分布",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict[str, float | str | int]]) -> None:
    fieldnames = ["layer", "group", "count", "min", "max", "std", "p99_abs", "p99.9_abs", "clip_rate_abs_gt_1", "adaptive_absmax"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_distributions(stats: Dict[str, ActivationStats], rows_by_layer: Dict[str, Dict[str, float | str | int]], plot_layers: List[str], out_dir: Path, dac_bits: int) -> None:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "distributions"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for layer_name in plot_layers:
        if layer_name not in stats:
            print(f"[warn] plot layer not found: {layer_name}")
            continue
        raw_np = stats[layer_name].sample_array()
        if raw_np.size == 0:
            continue
        raw = torch.from_numpy(raw_np)
        fixed_clip = raw.clamp(-1.0, 1.0)
        scale = max(float(rows_by_layer[layer_name]["adaptive_absmax"]), 1e-12)
        adaptive = quantize_symmetric(raw / scale, dac_bits) * scale

        plt.figure(figsize=(8, 5))
        lo, hi = np.percentile(raw_np, [0.1, 99.9])
        bins = np.linspace(float(lo), float(hi), 120)
        plt.hist(raw_np, bins=bins, density=True, alpha=0.45, label="original activation")
        plt.hist(fixed_clip.numpy(), bins=bins, density=True, alpha=0.45, label="fixed [-1,1] clipped")
        plt.hist(adaptive.numpy(), bins=bins, density=True, alpha=0.45, label="layer-wise adaptive quantized")
        plt.axvline(-1.0, color="black", linewidth=1, linestyle="--")
        plt.axvline(1.0, color="black", linewidth=1, linestyle="--")
        plt.title(layer_name)
        plt.xlabel("activation value")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        safe_name = layer_name.replace(".", "_")
        plt.savefig(plot_dir / f"{safe_name}_distribution.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_model, crosssim_model, base_payload, crosssim_payload = load_baseline_and_crosssim(
        args.baseline_checkpoint,
        args.crosssim_checkpoint,
        device,
    )
    loader, _vocab = build_validation_loader_from_payload(
        base_payload,
        args.coco_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    mapped_layer_names = [name for name, _module in mapped_crosssim_linears(crosssim_model)]
    stats = {name: ActivationStats(layer_name=name, sample_limit=args.sample_limit) for name in mapped_layer_names}

    handles = []
    source_model = base_model if args.activation_source == "baseline" else crosssim_model
    source_modules = dict(source_model.named_modules())
    for layer_name in mapped_layer_names:
        if layer_name not in source_modules:
            raise KeyError(f"Cannot find activation source layer: {layer_name}")
        module = source_modules[layer_name]

        def make_hook(name: str):
            def hook(_module, inputs, _output):
                stats[name].update(inputs[0])
            return hook
        handles.append(module.register_forward_hook(make_hook(layer_name)))

    try:
        processed = run_teacher_forcing_batches(source_model, loader, device, args.max_batches)
    finally:
        for handle in handles:
            handle.remove()

    rows = [item.row() for item in stats.values()]
    rows.sort(key=lambda row: str(row["layer"]))
    rows_by_layer = {str(row["layer"]): row for row in rows}
    write_csv(args.output_dir / "activation_clip_stats.csv", rows)

    with (args.output_dir / "activation_clip_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_checkpoint": str(args.baseline_checkpoint),
                "crosssim_checkpoint": str(args.crosssim_checkpoint),
                "crosssim_condition": crosssim_payload.get("condition"),
                "activation_source": args.activation_source,
                "processed_images": processed,
                "max_batches": args.max_batches,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_layers = [item.strip() for item in args.plot_layers.split(",") if item.strip()]
    plot_distributions(stats, rows_by_layer, plot_layers, args.output_dir, args.dac_bits)

    print(f"Processed images: {processed}")
    print(f"Saved stats: {args.output_dir / 'activation_clip_stats.csv'}")
    print(f"Saved plots: {args.output_dir / 'distributions'}")


if __name__ == "__main__":
    main()
