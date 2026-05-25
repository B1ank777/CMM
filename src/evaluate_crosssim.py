from __future__ import annotations

# 标准库
import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

# 深度学习框架
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# COCO 数据管线
from .coco_preprocess.coco_io import load_coco_captions
from .coco_preprocess.dataset import CocoCaptionDataset, collate_fn
from .coco_preprocess.loader import DEFAULT_COCO_ROOT, default_image_transform
from .coco_preprocess.tokenizer import WordTokenizer
from .coco_preprocess.vocab import Vocabulary

# CrossSim 模型映射工具
from .map_crosssim import (
    build_model_from_crosssim_payload,
    build_model_from_payload,
    load_checkpoint,
    synchronize_crosssim_cores,
)
from .map_cmm import build_model_from_cmm_payload

# 运行示例：
#   # 评估 CrossSim 映射模型 vs 数字基线（默认 100 batch）
#   python -m src.evaluate_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt --crosssim-checkpoint checkpoints/caption_transformer_crosssim.pt
#        
#   # 限制评估 batch 数，使用验证集子集以加快评估
#   python -m src.evaluate_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt
#       --crosssim-checkpoint checkpoints/crosssim_gpu.pt \
#       --max-batches 50 --subset-size 500
#
#   # CPU 评估，指定 COCO 根目录
#   python -m src.evaluate_crosssim \
#       --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --crosssim-checkpoint checkpoints/caption_transformer_crosssim.pt \
#       --device cpu --coco-root /data/coco


def parse_args() -> argparse.Namespace:
    """解析基线模型与 CrossSim 模型对比评估参数。"""
    parser = argparse.ArgumentParser(description="Evaluate baseline vs CrossSim model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Baseline training checkpoint")
    parser.add_argument("--crosssim-checkpoint", type=Path, required=True, help="CrossSim checkpoint")
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
    """优先从 checkpoint 恢复词表；缺失时从 COCO 训练标注重建。"""
    tokenizer = WordTokenizer()
    vocab = Vocabulary(tokenizer=tokenizer, min_freq=min_freq)

    # 若 checkpoint 中包含词表映射，直接恢复
    if "vocab_stoi" in payload and payload["vocab_stoi"]:
        stoi = payload["vocab_stoi"]
        vocab.stoi = stoi
        vocab.itos = {i: w for w, i in stoi.items()}
        return vocab

    # 否则从 COCO 训练集标注构建词表
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
    """构建 COCO val2014 DataLoader。"""
    dataset = CocoCaptionDataset(
        image_dir=str(coco_root / "val2014"),
        annotation_file=str(coco_root / "annotations" / "captions_val2014.json"),
        vocab=vocab,
        transform=default_image_transform(224),
        max_len=max_len,
    )

    # 可选：仅使用验证集的子集以加快评估
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
    """计算 loss 与 token accuracy（teacher forcing 模式）。"""
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

        # Teacher forcing：输入去掉最后一个 token，目标去掉第一个 token
        input_tokens = captions[:, :-1]
        target_tokens = captions[:, 1:]
        logits = model(images, input_tokens)
        # 展平为 (B*L, V) × (B*L) 计算交叉熵
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))

        # 按 token 级别统计准确率，排除 padding 位置
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
    crosssim_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[float, float, float]:
    """比较数字模型与 CrossSim 模型的 logits 差异。

    返回 (MAE, Max Error, RMSE) 三个指标，衡量忆阻器映射带来的输出偏差。
    """
    base_model.eval()
    crosssim_model.eval()

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

        # 分别前向传播，比较 logits 差异
        logits_base = base_model(images, input_tokens)
        logits_crosssim = crosssim_model(images, input_tokens)

        # 统计逐元素差异：平均绝对误差 / 最大误差 / 均方根误差
        diff = logits_base - logits_crosssim
        abs_diff = diff.abs()
        total_mae += abs_diff.mean().item()
        total_mse += diff.square().mean().item()
        max_error = max(max_error, abs_diff.max().item())
        steps += 1

    mean_mae = total_mae / max(steps, 1)
    mean_rmse = (total_mse / max(steps, 1)) ** 0.5
    return mean_mae, max_error, mean_rmse


def load_baseline_model(checkpoint: Path, device: torch.device):
    """加载原始数字基线模型（原始 nn.Linear，未经忆阻器映射）。"""
    payload, state = load_checkpoint(checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def load_crosssim_model(crosssim_checkpoint: Path, baseline_model: nn.Module, device: torch.device):
    """加载 CrossSim 或 CMM checkpoint，并返回可推理模型。

    CrossSim 模型内部维护了与 weight 分离的器件电导状态，
    load_state_dict 后必须调用 synchronize 将电导写回到模拟阵列中。
    CMM checkpoint 直接保存 r_pos/r_neg 等器件状态 buffer，无需额外同步。
    """
    crosssim_payload = torch.load(crosssim_checkpoint, map_location="cpu")
    is_cmm_checkpoint = crosssim_payload.get("format") == "cmm_v1" or "cmm_model_state_dict" in crosssim_payload
    if is_cmm_checkpoint:
        cmm_args = crosssim_payload.get("cmm_args")
        if not cmm_args:
            raise ValueError("CMM checkpoint missing 'cmm_args'.")
        if "cmm_model_state_dict" not in crosssim_payload:
            raise ValueError("CMM checkpoint missing 'cmm_model_state_dict'.")
        cmm_model = build_model_from_cmm_payload(baseline_model, cmm_args, device)
        cmm_model.load_state_dict(crosssim_payload["cmm_model_state_dict"])
        cmm_model.to(device)
        cmm_model.eval()
        return cmm_model, crosssim_payload

    crosssim_args = crosssim_payload.get("crosssim_args")
    if not crosssim_args:
        raise ValueError("Checkpoint missing 'crosssim_args' or 'cmm_args'.")
    if "crosssim_model_state_dict" not in crosssim_payload:
        raise ValueError("Checkpoint missing 'crosssim_model_state_dict' or 'cmm_model_state_dict'.")

    # 以基线模型结构为模板，构建 CrossSim 版本的模型
    crosssim_model = build_model_from_crosssim_payload(baseline_model, crosssim_args, device)
    crosssim_model.load_state_dict(crosssim_payload["crosssim_model_state_dict"])
    # 将加载的权重同步到 CrossSim 内部器件阵列
    synchronize_crosssim_cores(crosssim_model)
    crosssim_model.to(device)
    crosssim_model.eval()
    return crosssim_model, crosssim_payload


def main() -> None:
    """主流程：加载模型、跑验证集、打印 CrossSim 对比指标。"""
    args = parse_args()
    device = torch.device(args.device)

    # 加载基线模型与 CrossSim 模型
    base_model, base_payload = load_baseline_model(args.checkpoint, device)
    crosssim_model, crosssim_payload = load_crosssim_model(args.crosssim_checkpoint, base_model, device)

    # 构建词表与验证集 DataLoader
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

    # 忽略 padding token 的交叉熵损失
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    # 分别评估两个模型的 loss 与准确率
    base_loss, base_acc = evaluate_single_model(
        base_model, loader, criterion, device, args.max_batches, vocab.pad_id
    )
    crosssim_loss, crosssim_acc = evaluate_single_model(
        crosssim_model, loader, criterion, device, args.max_batches, vocab.pad_id
    )
    # 比较两模型的 logits 输出差异
    logit_mae, logit_max_error, logit_rmse = compare_logits(
        base_model, crosssim_model, loader, device, args.max_batches
    )

    # 打印评估结果汇总
    print("=== Evaluation Summary ===")
    print(f"Device: {device}")
    # 打印 checkpoint 中实际记录的映射参数，避免评估时误判实验条件。
    if crosssim_payload.get("format") == "cmm_v1" or "cmm_args" in crosssim_payload:
        cmm_args = crosssim_payload.get("cmm_args", {})
        mapped_name = "CMM"
        print(f"CMM format: {crosssim_payload.get('format', 'legacy-cmm')}")
        print(f"CMMLinear layers: {crosssim_payload.get('num_cmm_linear', 'unknown')}")
        print(f"CMM scope: {cmm_args.get('mapping_scope', 'decoder_only')}")
        print(f"CMM tile shape: {tuple(cmm_args.get('tile_shape', (128, 128)))}")
        print(f"CMM Rmin/Rmax: {cmm_args.get('rmin', 1e3)}/{cmm_args.get('rmax', 1e5)}")
        print(f"CMM write/read noise std: {cmm_args.get('write_noise_std', 0.0)}/{cmm_args.get('read_noise_std', 0.0)}")
    else:
        crosssim_args = crosssim_payload.get("crosssim_args", {})
        mapped_name = "CrossSim"
        print(f"CrossSim scope: {crosssim_args.get('mapping_scope', 'decoder_only')}")
        print(f"CrossSim tile shape: {tuple(crosssim_args.get('tile_shape', (128, 128)))}")
        print(f"CrossSim ADC/DAC: {crosssim_args.get('adc_resolution', 0)}/{crosssim_args.get('dac_resolution', 0)}")
        print(f"CrossSim GPU: {crosssim_args.get('use_gpu', False)}")
    print(f"Batches evaluated: {args.max_batches}")
    print(f"Baseline  loss: {base_loss:.6f}, token_acc: {base_acc:.6f}")
    print(f"{mapped_name}  loss: {crosssim_loss:.6f}, token_acc: {crosssim_acc:.6f}")
    print(f"Delta loss ({mapped_name.lower()} - base): {crosssim_loss - base_loss:.6f}")
    print(f"Delta acc  ({mapped_name.lower()} - base): {crosssim_acc - base_acc:.6f}")
    print(f"Logit MAE: {logit_mae:.6f}")
    print(f"Logit Max Error: {logit_max_error:.6f}")
    print(f"Logit RMSE: {logit_rmse:.6f}")


if __name__ == "__main__":
    main()
