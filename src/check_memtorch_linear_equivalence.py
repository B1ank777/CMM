from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coco_preprocess.loader import DEFAULT_COCO_ROOT
from .evaluate_memtorch import build_val_loader, build_vocab_from_payload_or_data
from .map_memtorch import build_model_from_payload, load_checkpoint, patch_module_memristive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare each baseline nn.Linear with an isolated MemTorch bh.Linear."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Baseline training checkpoint")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument(
        "--scope",
        type=str,
        default="self_attn_only",
        choices=["self_attn_only", "cross_attn_only", "ffn_only", "all"],
        help="Which decoder Linear group to test.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means test all matched Linear layers")
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--adc-resolution", type=int, default=8)
    parser.add_argument("--max-input-voltage", type=float, default=0.3)
    parser.add_argument("--ron", type=float, default=1e2)
    parser.add_argument("--roff", type=float, default=1e4)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def matches_scope(name: str, scope: str) -> bool:
    if scope in {"self_attn_only", "all"}:
        if ".self_attn." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return True
    if scope in {"cross_attn_only", "all"}:
        if ".cross_attn." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return True
    if scope in {"ffn_only", "all"}:
        if ".ffn." in name and name.rsplit(".", 1)[-1] in {"0", "3"}:
            return True
    return False


def collect_target_linears(model: nn.Module, scope: str, max_layers: int) -> List[Tuple[str, nn.Linear]]:
    targets: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and matches_scope(name, scope):
            targets.append((name, module))

    if max_layers > 0:
        targets = targets[:max_layers]
    if not targets:
        raise ValueError(f"No nn.Linear layers matched scope={scope}.")
    return targets


def capture_linear_inputs(
    model: nn.Module,
    targets: Iterable[Tuple[str, nn.Linear]],
    images: torch.Tensor,
    captions: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    for name, module in targets:
        def _hook(_module, inputs, layer_name=name):
            # 只捕获每层第一次真实前向输入，避免保存不必要的计算图。
            if layer_name not in captured:
                captured[layer_name] = inputs[0].detach()

        handles.append(module.register_forward_pre_hook(_hook))

    try:
        with torch.no_grad():
            model(images, captions[:, :-1])
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, _ in targets if name not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture inputs for: {missing}")
    return captured


def find_single_memtorch_linear(module: nn.Module) -> nn.Module:
    for child in module.modules():
        if type(child).__name__ == "Linear" and type(child).__module__.startswith("memtorch"):
            return child
    raise RuntimeError("No MemTorch Linear found after patching.")


def build_memtorch_linear(linear: nn.Linear, args: argparse.Namespace, device: torch.device) -> nn.Module:
    # 用 Sequential 包一层，复用正式 mapping 函数，保证单层检查和生产映射路径一致。
    container = nn.Sequential(copy.deepcopy(linear).cpu()).eval()
    mem_container = patch_module_memristive(
        container,
        use_bindings=False,
        tile_shape=(args.tile_rows, args.tile_cols),
        max_input_voltage=args.max_input_voltage,
        adc_resolution=args.adc_resolution,
        ron=args.ron,
        roff=args.roff,
    )
    mem_container.to(device).eval()
    return find_single_memtorch_linear(mem_container)


@torch.no_grad()
def compare_one_layer(
    name: str,
    base_linear: nn.Linear,
    x: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    base_linear.eval()
    mem_linear = build_memtorch_linear(base_linear, args, device)

    y_base = base_linear(x)
    y_mem = mem_linear(x)
    y_mem = y_mem.to(y_base.device)

    diff = y_base - y_mem
    abs_diff = diff.abs()
    flat_base = y_base.reshape(-1).float()
    flat_mem = y_mem.reshape(-1).float()

    if flat_base.norm().item() == 0.0 or flat_mem.norm().item() == 0.0:
        cosine = float("nan")
    else:
        cosine = F.cosine_similarity(flat_base.unsqueeze(0), flat_mem.unsqueeze(0), dim=1).item()

    return {
        "layer": name,
        "input_shape": tuple(x.shape),
        "output_shape": tuple(y_base.shape),
        "mae": abs_diff.mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_error": abs_diff.max().item(),
        "cosine": cosine,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "input_shape", "output_shape", "mae", "rmse", "max_error", "cosine"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: List[Dict[str, object]]) -> None:
    print("layer,input_shape,output_shape,mae,rmse,max_error,cosine")
    for row in rows:
        print(
            f"{row['layer']},{row['input_shape']},{row['output_shape']},"
            f"{row['mae']:.8f},{row['rmse']:.8f},{row['max_error']:.8f},{row['cosine']:.8f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    vocab = build_vocab_from_payload_or_data(payload, args.coco_root)
    max_len = int(payload["model_config"]["max_len"])
    loader = build_val_loader(
        coco_root=args.coco_root,
        vocab=vocab,
        max_len=max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )
    images, captions = next(iter(loader))
    images = images.to(device, non_blocking=True)
    captions = captions.to(device, non_blocking=True)

    targets = collect_target_linears(model, args.scope, args.max_layers)
    captured = capture_linear_inputs(model, targets, images, captions)

    print("=== MemTorch Linear Equivalence Check ===")
    print(f"Scope: {args.scope}")
    print(f"Device: {device}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    print(f"ADC resolution: {args.adc_resolution}")
    print(f"Layers tested: {len(targets)}")

    rows = [
        compare_one_layer(name, linear, captured[name], args, device)
        for name, linear in targets
    ]
    print_rows(rows)

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
