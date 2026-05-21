from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .coco_preprocess.coco_io import load_coco_captions
from .coco_preprocess.dataset import CocoCaptionDataset, collate_fn
from .coco_preprocess.loader import DEFAULT_COCO_ROOT, default_image_transform
from .coco_preprocess.tokenizer import WordTokenizer
from .coco_preprocess.vocab import Vocabulary
from .map_memtorch import build_memristive_model, build_model_from_payload, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs MemTorch model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Baseline training checkpoint")
    parser.add_argument("--mem-checkpoint", type=Path, required=True, help="MemTorch checkpoint")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--subset-size", type=int, default=0, help="0 means full validation set")
    return parser.parse_args()


def build_vocab_from_payload_or_data(
    payload: Dict[str, Any],
    coco_root: Path,
    min_freq: int = 5,
) -> Vocabulary:
    tokenizer = WordTokenizer()
    vocab = Vocabulary(tokenizer=tokenizer, min_freq=min_freq)

    if "vocab_stoi" in payload and payload["vocab_stoi"]:
        stoi = payload["vocab_stoi"]
        vocab.stoi = stoi
        vocab.itos = {i: w for w, i in stoi.items()}
        return vocab

    _, train_caps = load_coco_captions(str(coco_root / "annotations" / "captions_train2014.json"))
    vocab.build(train_caps)
    return vocab


def build_val_loader(
    coco_root: Path,
    vocab: Vocabulary,
    max_len: int,
    batch_size: int,
    num_workers: int,
    subset_size: int,
) -> DataLoader:
    dataset = CocoCaptionDataset(
        image_dir=str(coco_root / "val2014"),
        annotation_file=str(coco_root / "annotations" / "captions_val2014.json"),
        vocab=vocab,
        transform=default_image_transform(224),
        max_len=max_len,
    )

    if subset_size > 0:
        subset_size = min(subset_size, len(dataset))
        dataset = Subset(dataset, range(subset_size))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_fn, pad_id=vocab.pad_id),
    )


@torch.no_grad()
def evaluate_single_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
    pad_id: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    processed_batches = 0

    for batch_idx, (images, captions) in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        if batch_idx > max_batches:
            break

        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        input_tokens = captions[:, :-1]
        target_tokens = captions[:, 1:]
        logits = model(images, input_tokens)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))

        preds = logits.argmax(dim=-1)
        valid_mask = target_tokens.ne(pad_id)
        correct = preds.eq(target_tokens) & valid_mask

        total_loss += loss.item()
        total_correct += correct.sum().item()
        total_tokens += valid_mask.sum().item()
        processed_batches += 1

    mean_loss = total_loss / max(processed_batches, 1)
    token_acc = total_correct / max(total_tokens, 1)
    return mean_loss, token_acc


@torch.no_grad()
def compare_logits(
    base_model: nn.Module,
    mem_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[float, float, float]:
    base_model.eval()
    mem_model.eval()

    total_mae = 0.0
    total_mse = 0.0
    max_error = 0.0
    steps = 0

    for batch_idx, (images, captions) in enumerate(tqdm(loader, desc="logit-compare", leave=False), start=1):
        if batch_idx > max_batches:
            break

        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        input_tokens = captions[:, :-1]

        logits_base = base_model(images, input_tokens)
        logits_mem = mem_model(images, input_tokens)

        diff = logits_base - logits_mem
        abs_diff = diff.abs()
        total_mae += abs_diff.mean().item()
        total_mse += diff.square().mean().item()
        max_error = max(max_error, abs_diff.max().item())
        steps += 1

    mean_mae = total_mae / max(steps, 1)
    mean_rmse = (total_mse / max(steps, 1)) ** 0.5
    return mean_mae, max_error, mean_rmse


def load_baseline_model(checkpoint: Path, device: torch.device):
    payload, state = load_checkpoint(checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def load_mem_model(mem_checkpoint: Path, baseline_model: nn.Module, device: torch.device):
    mem_payload = torch.load(mem_checkpoint, map_location="cpu")
    mem_args = mem_payload.get("memtorch_args", {})
    mapping_scope = mem_args.get("mapping_scope", "decoder_only")

    if bool(mem_args.get("use_bindings", False)):
        print(f"Info: loading {mem_checkpoint.name} with use_bindings=False for inference compatibility.")

    mem_model = build_memristive_model(
        model=baseline_model,
        use_bindings=False,
        scope=mapping_scope,
        tile_shape=tuple(mem_args.get("tile_shape", (128, 128))),
        max_input_voltage=float(mem_args.get("max_input_voltage", 0.3)),
        adc_resolution=int(mem_args.get("adc_resolution", 8)),
        ron=float(mem_args.get("r_on", 1e2)),
        roff=float(mem_args.get("r_off", 1e4)),
    )
    mem_model.load_state_dict(mem_payload["mem_model_state_dict"])
    mem_model.to(device)
    mem_model.eval()
    return mem_model, mem_payload


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    base_model, base_payload = load_baseline_model(args.checkpoint, device)
    mem_model, mem_payload = load_mem_model(args.mem_checkpoint, base_model, device)

    vocab = build_vocab_from_payload_or_data(base_payload, args.coco_root)
    max_len = int(base_payload["model_config"]["max_len"])

    loader = build_val_loader(
        coco_root=args.coco_root,
        vocab=vocab,
        max_len=max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    base_loss, base_acc = evaluate_single_model(
        base_model, loader, criterion, device, args.max_batches, vocab.pad_id
    )
    mem_loss, mem_acc = evaluate_single_model(
        mem_model, loader, criterion, device, args.max_batches, vocab.pad_id
    )
    logit_mae, logit_max_error, logit_rmse = compare_logits(
        base_model, mem_model, loader, device, args.max_batches
    )

    print("=== Evaluation Summary ===")
    print(f"Device: {device}")
    mem_args = mem_payload.get("memtorch_args", {})
    # 打印实际从 mem checkpoint 读取到的映射配置，避免误以为评估阶段仍在使用默认参数。
    print(f"Mem scope: {mem_args.get('mapping_scope', 'decoder_only')}")
    print(f"Mem tile shape: {tuple(mem_args.get('tile_shape', (128, 128)))}")
    print(f"Mem ADC resolution: {mem_args.get('adc_resolution', 8)}")
    print(f"Mem max input voltage: {mem_args.get('max_input_voltage', 0.3)}")
    print(f"Batches evaluated: {args.max_batches}")
    print(f"Baseline  loss: {base_loss:.6f}, token_acc: {base_acc:.6f}")
    print(f"MemTorch  loss: {mem_loss:.6f}, token_acc: {mem_acc:.6f}")
    print(f"Delta loss (mem - base): {mem_loss - base_loss:.6f}")
    print(f"Delta acc  (mem - base): {mem_acc - base_acc:.6f}")
    print(f"Logit MAE: {logit_mae:.6f}")
    print(f"Logit Max Error: {logit_max_error:.6f}")
    print(f"Logit RMSE: {logit_rmse:.6f}")


if __name__ == "__main__":
    main()
