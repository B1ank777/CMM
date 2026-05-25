# 单张图片描述生成脚本：加载训练好的模型，对输入图片生成文字描述
#
# 使用方式:
#   python -m src.generate_caption --checkpoint <检查点路径> --image <图片路径>
#
# 示例:
#   python -m src.generate_caption --checkpoint checkpoints/best.pt --image test.jpg
#   python -m src.generate_caption --checkpoint checkpoints/best.pt --image test.jpg --device cpu --max-len 50

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

from .coco_preprocess.loader import default_image_transform
from .coco_preprocess.tokenizer import WordTokenizer
from .coco_preprocess.vocab import Vocabulary
from .map_crosssim import (
    build_model_from_crosssim_payload,
    build_model_from_payload,
    synchronize_crosssim_cores,
)
from .map_cmm import build_model_from_cmm_payload


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Generate caption from a single image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点路径")
    parser.add_argument("--image", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--max-len", type=int, default=30, help="生成描述的最大长度")
    return parser.parse_args()


def build_vocab_from_payload(payload: Dict[str, Any]) -> Vocabulary:
    """从检查点数据中恢复词表"""
    tokenizer = WordTokenizer()
    vocab = Vocabulary(tokenizer=tokenizer, min_freq=1)

    stoi = payload.get("vocab_stoi")
    if not stoi:
        raise ValueError("Checkpoint payload missing vocab_stoi.")

    vocab.stoi = stoi
    vocab.itos = {i: w for w, i in stoi.items()}  # 反向索引：id -> 词
    return vocab


@torch.no_grad()
def generate_caption(model, image, vocab, device, max_len=30):
    """使用模型对图片生成描述文本"""
    model.eval()

    bos_id = vocab.stoi["<bos>"]  # 起始标记
    eos_id = vocab.stoi["<eos>"]  # 结束标记

    image = image.to(device)

    tokens = model.generate(  # 自回归生成 token 序列
        image=image,
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=max_len,
    )

    return vocab.decode(tokens.cpu().tolist())  # 将 token id 序列解码为文本


def load_image_tensor(image_path: Path) -> torch.Tensor:
    """加载图片并转换为模型输入张量"""
    image = Image.open(image_path).convert("RGB")
    transform = default_image_transform(image_size=224)
    return transform(image)


def main() -> None:
    """主流程：加载模型 -> 读取图片 -> 生成描述"""
    args = parse_args()
    device = torch.device(args.device)

    # 加载检查点
    payload = torch.load(args.checkpoint, map_location="cpu")

    # 恢复词表
    vocab = build_vocab_from_payload(payload)

    # 恢复模型 —— 兼容标准训练检查点、CrossSim 检查点与 CMM 检查点。
    state_dict = payload.get("model_state_dict")
    if state_dict is not None:
        # 标准训练检查点：直接加载权重到 CaptionTransformer
        model = build_model_from_payload(payload)  # 从配置重建模型结构
        model.load_state_dict(state_dict)           # 加载训练权重
    elif "crosssim_model_state_dict" in payload:
        # CrossSim 检查点：先重建 analog 层结构，再加载权重并同步内部阵列。
        base_model = build_model_from_payload(payload)
        base_model.to(device)
        model = build_model_from_crosssim_payload(
            baseline_model=base_model,
            crosssim_args=payload.get("crosssim_args", {}),
            device=device,
        )
        model.load_state_dict(payload["crosssim_model_state_dict"])
        synchronize_crosssim_cores(model)
    elif payload.get("format") == "cmm_v1" or "cmm_model_state_dict" in payload:
        # CMM 检查点：先重建 CMMLinear 结构，再加载 r_pos/r_neg 等器件状态。
        if "cmm_model_state_dict" not in payload:
            raise ValueError("CMM checkpoint missing 'cmm_model_state_dict'.")
        base_model = build_model_from_payload(payload)
        base_model.to(device)
        model = build_model_from_cmm_payload(
            baseline_model=base_model,
            cmm_args=payload.get("cmm_args", {}),
            device=device,
        )
        model.load_state_dict(payload["cmm_model_state_dict"])
    else:
        raise ValueError(
            "Checkpoint payload missing both 'model_state_dict' (standard) "
            "'crosssim_model_state_dict' (CrossSim-mapped), and "
            "'cmm_model_state_dict' (CMM-mapped)."
        )

    model.to(device)  # 将模型移至目标设备

    # 读取图片并生成描述
    image_tensor = load_image_tensor(args.image)
    caption = generate_caption(
        model=model,
        image=image_tensor,
        vocab=vocab,
        device=device,
        max_len=args.max_len,
    )

    print(caption)


if __name__ == "__main__":
    main()
