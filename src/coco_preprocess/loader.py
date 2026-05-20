from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from torchvision import transforms

from .coco_io import load_coco_captions
from .dataset import CocoCaptionDataset, collate_fn
from .tokenizer import BaseTokenizer, WordTokenizer
from .vocab import Vocabulary


DEFAULT_COCO_ROOT = Path("data/coco")


def default_image_transform(image_size: int = 224):
    """返回默认的图像预处理变换：缩放 → 转张量 → ImageNet 标准化"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                std=[0.229, 0.224, 0.225],  # ImageNet 标准差
            ),
        ]
    )


def get_split_paths(split: str, coco_root: Path = DEFAULT_COCO_ROOT):
    """根据 split (train/val) 返回对应的图片目录和标注文件路径"""
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}. Use 'train' or 'val'.")

    image_dir = coco_root / ("train2014" if split == "train" else "val2014")
    annotation_file = coco_root / "annotations" / (
        "captions_train2014.json" if split == "train" else "captions_val2014.json"
    )
    return image_dir, annotation_file


def build_vocab_from_train(
    coco_root: Path = DEFAULT_COCO_ROOT,
    min_freq: int = 5,
    tokenizer: Optional[BaseTokenizer] = None,
) -> Vocabulary:
    """使用训练集标题构建词汇表（便捷函数）"""
    tokenizer = tokenizer or WordTokenizer()
    _, train_annotation = get_split_paths("train", coco_root)
    _, captions = load_coco_captions(str(train_annotation))

    vocab = Vocabulary(tokenizer=tokenizer, min_freq=min_freq)
    vocab.build(captions)
    return vocab


def build_dataloader(
    split: str,
    vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    max_len: int = 30,
    image_size: int = 224,
    coco_root: Path = DEFAULT_COCO_ROOT,
):
    """构建 COCO 标题的 DataLoader，训练集默认打乱，验证集默认不打乱"""
    image_dir, annotation_file = get_split_paths(split, coco_root)

    if shuffle is None:
        shuffle = split.lower() == "train"  # 训练集默认打乱

    dataset = CocoCaptionDataset(
        image_dir=str(image_dir),
        annotation_file=str(annotation_file),
        vocab=vocab,
        transform=default_image_transform(image_size=image_size),
        max_len=max_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id=vocab.pad_id),
    )
