from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .coco_io import load_coco_captions
from .vocab import Vocabulary


class CocoCaptionDataset(Dataset):
    """COCO 图片标题数据集：返回 (图像张量, 标题索引序列) 对"""

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        vocab: Vocabulary,
        transform: Optional[Callable] = None,
        max_len: int = 30,
    ):
        self.image_dir = Path(image_dir)
        self.samples, _ = load_coco_captions(annotation_file)
        self.vocab = vocab
        self.transform = transform  # 图像预处理变换
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_name, caption = self.samples[idx]
        image_path = self.image_dir / file_name

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        token_ids = self.vocab.encode(caption, max_len=self.max_len)
        return image, torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    """批次整理函数：将不等长标题填充到批次内最大长度"""
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # 计算批次内最大标题长度，用 pad_id 填充较短序列
    max_len = max(cap.size(0) for cap in captions)
    padded = torch.full((len(captions), max_len), fill_value=pad_id, dtype=torch.long)

    for i, cap in enumerate(captions):
        padded[i, : cap.size(0)] = cap

    return images, padded
