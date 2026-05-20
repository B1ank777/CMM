"""COCO 图像标题数据预处理演示脚本。

流程：检查数据路径 → 构建词汇表 → 创建训练/验证 DataLoader → 打印批次信息。
"""
import argparse
from pathlib import Path

from coco_preprocess.loader import (
    DEFAULT_COCO_ROOT,
    build_dataloader,
    build_vocab_from_train,
)
from coco_preprocess.tokenizer import WordTokenizer


def check_paths(coco_root: Path) -> None:
    """检查 COCO 数据集必需的目录和文件是否存在，缺失则抛出异常"""
    required = [
        coco_root / "train2014",
        coco_root / "val2014",
        coco_root / "annotations" / "captions_train2014.json",
        coco_root / "annotations" / "captions_val2014.json",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        lines = ["Required COCO paths are missing:"]
        lines.extend([f"- {p}" for p in missing])
        raise FileNotFoundError("\n".join(lines))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="COCO caption preprocessing demo")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    check_paths(args.coco_root)

    # 构建分词器与词汇表
    tokenizer = WordTokenizer()
    vocab = build_vocab_from_train(
        coco_root=args.coco_root,
        min_freq=args.min_freq,
        tokenizer=tokenizer,
    )

    print(f"Vocab size: {len(vocab.stoi)}")

    # 训练集 DataLoader（打乱）
    train_loader = build_dataloader(
        split="train",
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        image_size=args.image_size,
        coco_root=args.coco_root,
        shuffle=True,
    )

    # 验证集 DataLoader（不打乱）
    val_loader = build_dataloader(
        split="val",
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        image_size=args.image_size,
        coco_root=args.coco_root,
        shuffle=False,
    )

    # 取一个批次查看形状与解码示例
    train_images, train_captions = next(iter(train_loader))
    val_images, val_captions = next(iter(val_loader))

    print(f"Train batch images shape: {tuple(train_images.shape)}")
    print(f"Train batch captions shape: {tuple(train_captions.shape)}")
    print(f"Val batch images shape: {tuple(val_images.shape)}")
    print(f"Val batch captions shape: {tuple(val_captions.shape)}")
    print("Decoded sample:", vocab.decode(train_captions[0].tolist()))


if __name__ == "__main__":
    main()
