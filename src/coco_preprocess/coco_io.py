import json
from pathlib import Path
from typing import List, Tuple


def load_coco_captions(annotation_file: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """从 COCO 标注 JSON 文件中加载 (图片文件名, 标题) 对及全部标题列表"""
    annotation_path = Path(annotation_file)
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建 image_id → 文件名的映射
    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    samples: List[Tuple[str, str]] = []
    captions: List[str] = []

    for ann in data["annotations"]:
        caption = ann["caption"]
        image_id = ann["image_id"]
        file_name = image_id_to_file[image_id]
        samples.append((file_name, caption))
        captions.append(caption)

    return samples, captions
