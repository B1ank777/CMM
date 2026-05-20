"""
COCO 图像描述生成训练脚本。

使用 CaptionTransformer（ResNet 编码器 + Transformer 解码器）在 COCO Captions 数据集上训练。
支持混合精度（AMP）、梯度累积、梯度裁剪等训练技巧，默认配置适合 8GB 显存 GPU。

启动训练:
    python -m src.train_captioner

常用参数:
    --coco-root PATH      COCO 数据集根目录（默认: data/coco）
    --epochs 20           训练轮数（默认: 10）
    --batch-size 32       批次大小（默认: 24）
    --lr 1e-4             学习率（默认: 1e-4）
    --train-cnn           解冻 CNN 编码器进行微调
    --no-amp              禁用混合精度
    --accum-steps 2       梯度累积步数（默认: 1）
    --save-dir PATH       检查点保存目录（默认: checkpoints）
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_preprocess.dataset import collate_fn
from coco_preprocess.loader import DEFAULT_COCO_ROOT, default_image_transform
from coco_preprocess.coco_io import load_coco_captions
from coco_preprocess.dataset import CocoCaptionDataset
from coco_preprocess.tokenizer import WordTokenizer
from coco_preprocess.vocab import Vocabulary
from models import CaptionTransformer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在 COCO Captions 上训练 Caption Transformer")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))

    # 模型配置（默认值适合 8GB 显存 GPU）
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--min-freq", type=int, default=5)

    # 训练配置
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--train-cnn", action="store_true", help="解冻编码器 CNN 进行微调")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def check_paths(coco_root: Path) -> None:
    """检查 COCO 数据集所需路径是否存在，缺失则抛出 FileNotFoundError。"""
    required = [
        coco_root / "train2014",
        coco_root / "annotations" / "captions_train2014.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        lines = ["COCO 训练所需路径缺失:"]
        lines.extend([f"- {p}" for p in missing])
        raise FileNotFoundError("\n".join(lines))


def detect_device() -> torch.device:
    """检测可用设备，优先使用 CUDA。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """设置随机种子以确保可复现性。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: CaptionTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    accum_steps: int,
    use_amp: bool,
) -> float:
    """执行一个训练 epoch。

    使用教师强制（teacher forcing）：输入为 caption[:, :-1]，目标为 caption[:, 1:]。
    支持混合精度和梯度累积。

    返回:
        平均 loss。
    """
    model.train()
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for step, (images, captions) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        # 教师强制：输入去掉最后一个 token，目标去掉第一个 token
        input_tokens = captions[:, :-1]
        target_tokens = captions[:, 1:]

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, input_tokens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            loss = loss / accum_steps  # 归一化到单步

        # 梯度累积：每 accum_steps 步更新一次参数
        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        progress.set_postfix(loss=f"{total_loss / step:.4f}")

    return total_loss / len(loader)


def build_train_loader(
    coco_root: Path,
    vocab: Vocabulary,
    max_len: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """构建 COCO 训练数据的 DataLoader。

    返回的 DataLoader 会在 CPU 上做数据增强，并通过 pin_memory 加速 GPU 传输。
    """
    dataset = CocoCaptionDataset(
        image_dir=str(coco_root / "train2014"),
        annotation_file=str(coco_root / "annotations" / "captions_train2014.json"),
        vocab=vocab,
        transform=default_image_transform(image_size=224),
        max_len=max_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_fn, pad_id=vocab.pad_id),
    )


def save_checkpoint(
    save_dir: Path,
    epoch: int,
    model: CaptionTransformer,
    optimizer: torch.optim.Optimizer,
    vocab: Vocabulary,
    args: argparse.Namespace,
) -> None:
    """保存检查点，包含模型权重、优化器状态、词表和模型配置。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"caption_transformer_epoch_{epoch}.pt"

    payload: Dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "vocab_size": len(vocab.stoi),
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "num_layers": args.num_layers,
            "max_len": args.max_len,
            "pad_id": vocab.pad_id,
        },
        "vocab_stoi": vocab.stoi,
    }
    torch.save(payload, ckpt_path)


def main() -> None:
    """训练入口：解析参数 → 准备数据 → 构建模型 → 训练循环 → 保存检查点。"""
    args = parse_args()
    check_paths(args.coco_root)
    set_seed(args.seed)

    device = detect_device()

    # 加载 COCO Captions 并构建词表
    annotation_file = args.coco_root / "annotations" / "captions_train2014.json"
    _, train_captions = load_coco_captions(str(annotation_file))

    tokenizer = WordTokenizer()
    vocab = Vocabulary(tokenizer=tokenizer, min_freq=args.min_freq)
    vocab.build(train_captions)

    # 构建数据加载器
    loader = build_train_loader(
        coco_root=args.coco_root,
        vocab=vocab,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 构建模型
    model = CaptionTransformer(
        vocab_size=len(vocab.stoi),
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
        encoder_type="resnet",
        pad_id=vocab.pad_id,
        train_cnn=args.train_cnn,
        pretrained_encoder=True,
    ).to(device)

    # 优化器 & 损失函数（忽略填充位置）
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    # 混合精度（仅 CUDA 可用时启用）
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    # 打印训练环境信息
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory (GB): {total_mem_gb:.2f}")
    print(f"Vocab size: {len(vocab.stoi)}")
    print(f"Train samples: {len(loader.dataset)}")
    print(f"AMP enabled: {use_amp}")

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
            use_amp=use_amp,
        )
        print(f"epoch={epoch}, loss={loss:.4f}")

        # 每 epoch 保存一次检查点
        save_checkpoint(
            save_dir=args.save_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            vocab=vocab,
            args=args,
        )

    # 保存训练配置以供复现
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
