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
# build_model_from_payload: 从检查点配置重建 CaptionTransformer 结构
# build_memristive_model: 将模型权重映射为忆阻器交叉阵列仿真模型
from .map_memtorch import build_memristive_model, build_model_from_payload


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Generate caption from a single image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点路径")
    parser.add_argument("--image", type=Path, required=True, help="输入图片路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--max-len", type=int, default=30, help="生成描述的最大长度")
    parser.add_argument(
        "--use-bindings",
        action="store_true",
        help="对 MemTorch 检查点显式启用 bindings。默认关闭，因为当前 memtorch 版本的 bindings 推理签名不稳定。",
    )
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

    # 恢复模型 —— 兼容两种检查点格式
    state_dict = payload.get("model_state_dict")
    if state_dict is not None:
        # 标准训练检查点：直接加载权重到 CaptionTransformer
        model = build_model_from_payload(payload)  # 从配置重建模型结构
        model.load_state_dict(state_dict)           # 加载训练权重
    elif "mem_model_state_dict" in payload:
        # 忆阻器映射后的检查点：需要先构建忆阻器模型结构，再加载权重
        model = build_model_from_payload(payload)
        mem_args = payload.get("memtorch_args", {})
        # state_dict 在 bindings / 非 bindings 两种变体间兼容。
        # 默认走 Python 推理路径，因为 memtorch 1.1.6 的 bindings 在生成时
        # 可能因 tiled_inference 签名不匹配而失败。
        use_bindings = args.use_bindings and bool(mem_args.get("use_bindings", False))
        if bool(mem_args.get("use_bindings", False)) and not use_bindings:
            print("Info: 加载 MemTorch 检查点时强制使用 use_bindings=False 以确保推理兼容性。")
        model = build_memristive_model(  # 构建忆阻器交叉阵列仿真模型
            model=model,
            use_bindings=use_bindings,
            scope=mem_args.get("mapping_scope", "decoder_only"),
            tile_shape=tuple(mem_args.get("tile_shape", (128, 128))),       # 交叉阵列分块大小
            max_input_voltage=float(mem_args.get("max_input_voltage", 0.3)), # 最大输入电压
            adc_resolution=int(mem_args.get("adc_resolution", 8)),           # ADC 分辨率
            ron=float(mem_args.get("r_on", 1e2)),                            # 低阻态
            roff=float(mem_args.get("r_off", 1e4)),                          # 高阻态
        )
        model.load_state_dict(payload["mem_model_state_dict"])  # 加载映射后的权重
    else:
        raise ValueError(
            "Checkpoint payload missing both 'model_state_dict' (standard) "
            "and 'mem_model_state_dict' (memristor-mapped)."
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
