from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from .coco_preprocess.loader import DEFAULT_COCO_ROOT
from .compute_metrics_pycoco import (
    build_predictions_json,
    build_vocab,
    collect_image_ids,
    evaluate_with_pycoco,
    require_pycoco,
)
from .dac_adaptive_utils import (
    build_validation_loader_from_payload,
    load_baseline_and_crosssim,
    mapped_crosssim_linears,
    patch_adaptive_dac,
    restore_patched_forwards,
    run_teacher_forcing_batches,
)


# 运行示例：
#   python -m src.evaluate_cmm_crosssim_adaptive_dac ^
#       --baseline-checkpoint checkpoints\caption_transformer_epoch_10.pt ^
#       --crosssim-checkpoint checkpoints\cmm_crosssim_dac_conditions\caption_transformer_dac-12bit_cmm_crosssim.pt ^
#       --output experiments\dac_range_root_cause\adaptive_dac12_metrics.json ^
#       --device cuda --limit 1000 --calibration-batches 50


DEFAULT_BASELINE = Path("checkpoints/caption_transformer_epoch_10.pt")
DEFAULT_CMM_CROSSSIM_DAC12 = Path(
    "checkpoints/cmm_crosssim_dac_conditions/caption_transformer_dac-12bit_cmm_crosssim.pt"
)
DEFAULT_OUTPUT = Path("experiments/dac_range_root_cause/adaptive_dac12_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current vs adaptive DAC-12 CMM-CrossSim models.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE, help="数字基线 checkpoint")
    parser.add_argument("--crosssim-checkpoint", type=Path, default=DEFAULT_CMM_CROSSSIM_DAC12, help="DAC-12 CMM-CrossSim checkpoint")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT, help="COCO 数据根目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="指标 JSON 输出路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--max-len", type=int, default=30, help="生成描述最大长度")
    parser.add_argument("--limit", type=int, default=1000, help="pycocoevalcap 图片数量")
    parser.add_argument("--batch-size", type=int, default=32, help="生成 caption 的 batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--calibration-batches", type=int, default=50, help="layer-wise adaptive DAC 的校准 batch 数")
    parser.add_argument("--calibration-batch-size", type=int, default=8, help="校准 activation 统计 batch size")
    parser.add_argument("--subset-size", type=int, default=0, help="校准 DataLoader 子集大小；0 表示不截断")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过数字 baseline，只评测三个 DAC 设置")
    return parser.parse_args()


def collect_layer_absmax(model, layer_names: List[str], loader, device: torch.device, max_batches: int) -> Dict[str, float]:
    """用真实 activation 统计每层最大绝对值，作为 layer-wise DAC range。"""
    scales = {name: 0.0 for name in layer_names}
    source_modules = dict(model.named_modules())
    handles = []
    for layer_name in layer_names:
        if layer_name not in source_modules:
            raise KeyError(f"Cannot find calibration source layer: {layer_name}")
        module = source_modules[layer_name]

        def make_hook(name: str):
            def hook(_module, inputs, _output):
                value = float(inputs[0].detach().abs().amax().cpu())
                scales[name] = max(scales[name], value)
            return hook
        handles.append(module.register_forward_hook(make_hook(layer_name)))

    try:
        run_teacher_forcing_batches(model, loader, device, max_batches)
    finally:
        for handle in handles:
            handle.remove()

    return {name: max(value, 1e-12) for name, value in scales.items()}


def evaluate_named_model(
    name: str,
    model,
    vocab,
    coco_root: Path,
    image_ids: List[int],
    output_dir: Path,
    device: torch.device,
    max_len: int,
    batch_size: int,
    num_workers: int,
    COCO,
    COCOEvalCap,
) -> Dict[str, Any]:
    """生成 caption 并返回 COCO 指标。"""
    pred_file = output_dir / f"{name}_predictions.json"
    build_predictions_json(model, vocab, coco_root, image_ids, pred_file, device, max_len, batch_size, num_workers)
    metrics = evaluate_with_pycoco(
        COCO,
        COCOEvalCap,
        coco_root / "annotations" / "captions_val2014.json",
        pred_file,
        image_ids,
    )
    return {"model": name, "predictions": str(pred_file), "metrics": metrics}


def main() -> None:
    args = parse_args()
    COCO, COCOEvalCap = require_pycoco()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output.parent / "predictions"

    base_model, current_model, base_payload, _crosssim_payload = load_baseline_and_crosssim(
        args.baseline_checkpoint,
        args.crosssim_checkpoint,
        device,
    )
    vocab = build_vocab(base_payload)
    image_ids = collect_image_ids(args.coco_root / "annotations" / "captions_val2014.json", args.limit)

    rows: List[Dict[str, Any]] = []
    if not args.skip_baseline:
        rows.append(
            evaluate_named_model(
                "baseline",
                base_model,
                vocab,
                args.coco_root,
                image_ids,
                pred_dir,
                device,
                args.max_len,
                args.batch_size,
                args.num_workers,
                COCO,
                COCOEvalCap,
            )
        )

    rows.append(
        evaluate_named_model(
            "dac12_current_fixed_-1_1",
            current_model,
            vocab,
            args.coco_root,
            image_ids,
            pred_dir,
            device,
            args.max_len,
            args.batch_size,
            args.num_workers,
            COCO,
            COCOEvalCap,
        )
    )

    calibration_loader, _ = build_validation_loader_from_payload(
        base_payload,
        args.coco_root,
        batch_size=args.calibration_batch_size,
        num_workers=0,
        subset_size=args.subset_size,
    )
    mapped_layer_names = [name for name, _module in mapped_crosssim_linears(current_model)]
    layer_scales = collect_layer_absmax(base_model, mapped_layer_names, calibration_loader, device, args.calibration_batches)
    with (args.output.parent / "layer_wise_dac_scales.json").open("w", encoding="utf-8") as f:
        json.dump(layer_scales, f, ensure_ascii=False, indent=2)

    # layer-wise 与 batch-wise 都从同一个 checkpoint 重载，避免前一个 patch 污染后一个设置。
    _base_model_2, layer_model, _payload_2, _ = load_baseline_and_crosssim(args.baseline_checkpoint, args.crosssim_checkpoint, device)
    handles = patch_adaptive_dac(layer_model, mode="layer-wise", layer_scales=layer_scales)
    try:
        rows.append(
            evaluate_named_model(
                "dac12_layer_wise_adaptive",
                layer_model,
                vocab,
                args.coco_root,
                image_ids,
                pred_dir,
                device,
                args.max_len,
                args.batch_size,
                args.num_workers,
                COCO,
                COCOEvalCap,
            )
        )
    finally:
        restore_patched_forwards(handles)

    _base_model_3, batch_model, _payload_3, _ = load_baseline_and_crosssim(args.baseline_checkpoint, args.crosssim_checkpoint, device)
    handles = patch_adaptive_dac(batch_model, mode="batch-wise")
    try:
        rows.append(
            evaluate_named_model(
                "dac12_batch_wise_adaptive",
                batch_model,
                vocab,
                args.coco_root,
                image_ids,
                pred_dir,
                device,
                args.max_len,
                args.batch_size,
                args.num_workers,
                COCO,
                COCOEvalCap,
            )
        )
    finally:
        restore_patched_forwards(handles)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_checkpoint": str(args.baseline_checkpoint),
                "crosssim_checkpoint": str(args.crosssim_checkpoint),
                "num_images": len(image_ids),
                "results": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved metrics: {args.output}")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['model']}: BLEU-4={metrics.get('Bleu_4', float('nan')):.4f}, "
            f"CIDEr={metrics.get('CIDEr', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
