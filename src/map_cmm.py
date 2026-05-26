from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .cmm import convert_module_to_cmm, count_cmm_linear_layers
from .map_crosssim import (
    build_model_from_payload,
    count_linear_layers_by_scope,
    freeze_encoder,
    load_checkpoint,
)

# 运行示例：
#   # 默认论文式 CMM 映射，范围沿用 decoder_only：decoder layers + output_proj
#   python -m src.map_cmm --checkpoint checkpoints/caption_transformer_epoch_10.pt
#
#   # 添加写入/读出噪声并保存到指定路径
#   python -m src.map_cmm --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --write-noise-std 0.001 --read-noise-std 0.0001 \
#       --output checkpoints/caption_transformer_cmm_it7.pt
#
#   # 添加 DAC/ADC 量化
#   python -m src.map_cmm --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --dac-resolution 8 --adc-resolution 10 \
#       --output checkpoints/caption_transformer_cmm_adc10_dac8.pt
#
#   # 只映射 decoder layers，暂不映射 vocabulary projection
#   python -m src.map_cmm --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --scope layers_only --output checkpoints/caption_transformer_cmm_layers.pt


def parse_args() -> argparse.Namespace:
    """解析 CMM 映射脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="将 CaptionTransformer 的线性层映射为论文式 CMM 等效层")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练检查点路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/caption_transformer_cmm.pt"),
        help="CMM 映射后模型的输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="映射设备",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="映射范围：output_only=仅 LM head；layers_only=仅 decoder 层；decoder_only=全部 decoder 线性层",
    )
    parser.add_argument("--tile-rows", type=int, default=128, help="CMM 阵列最大行数，forward 会按该行数分块")
    parser.add_argument("--tile-cols", type=int, default=128, help="CMM 阵列最大列数，forward 会按该列数分块")
    parser.add_argument("--rmin", type=float, default=1e3, help="Ron，器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="Roff，器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元状态量化 bit 数；0 表示连续状态")
    parser.add_argument("--write-noise-std", type=float, default=0.0, help="写入噪声强度，作用在内部状态 r 上")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="读噪声强度，作用在推理时 Rmem 上")
    parser.add_argument("--adc-resolution", type=int, default=0, help="ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=0, help="DAC 分辨率；0 表示理想 DAC")
    return parser.parse_args()


def build_cmm_model(
    model: nn.Module,
    scope: str,
    tile_shape: Tuple[int, int] = (128, 128),
    rmin: float = 1e3,
    rmax: float = 1e5,
    cell_bits: int = 0,
    write_noise_std: float = 0.0,
    read_noise_std: float = 0.0,
    adc_resolution: int = 0,
    dac_resolution: int = 0,
) -> nn.Module:
    """按 scope 将 CaptionTransformer 的线性层映射为 CMMLinear。"""
    if tile_shape[0] <= 0 or tile_shape[1] <= 0:
        raise ValueError("tile_shape 的行列数必须为正整数。")

    # 深拷贝避免影响原始模型
    cmm_model = copy.deepcopy(model)
    if not hasattr(cmm_model, "layers") or not hasattr(cmm_model, "output_proj"):
        raise ValueError("build_cmm_model expects a CaptionTransformer-like model.")

    # 按 scope 选择性替换 decoder layers 和/或 output_proj
    if scope in {"layers_only", "decoder_only"}:
        cmm_model.layers = convert_module_to_cmm(
            cmm_model.layers,
            rmin=rmin,
            rmax=rmax,
            cell_bits=cell_bits,
            write_noise_std=write_noise_std,
            read_noise_std=read_noise_std,
            tile_rows=tile_shape[0],
            tile_cols=tile_shape[1],
            adc_resolution=adc_resolution,
            dac_resolution=dac_resolution,
        )
    if scope in {"output_only", "decoder_only"}:
        cmm_model.output_proj = convert_module_to_cmm(
            cmm_model.output_proj,
            rmin=rmin,
            rmax=rmax,
            cell_bits=cell_bits,
            write_noise_std=write_noise_std,
            read_noise_std=read_noise_std,
            tile_rows=tile_shape[0],
            tile_cols=tile_shape[1],
            adc_resolution=adc_resolution,
            dac_resolution=dac_resolution,
        )
    return cmm_model


def make_cmm_args(args: argparse.Namespace) -> Dict[str, Any]:
    """整理保存到 checkpoint 的 CMM 配置元数据。"""
    return {
        "tile_shape": (args.tile_rows, args.tile_cols),
        "mapping_scope": args.scope,
        "rmin": args.rmin,
        "rmax": args.rmax,
        "cell_bits": args.cell_bits,
        "write_noise_std": args.write_noise_std,
        "read_noise_std": args.read_noise_std,
        "adc_resolution": getattr(args, "adc_resolution", 0),
        "dac_resolution": getattr(args, "dac_resolution", 0),
    }


def build_model_from_cmm_payload(
    baseline_model: nn.Module,
    cmm_args: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """根据 CMM checkpoint 元数据重建 CMM 模型结构。"""
    base_for_mapping = baseline_model.to(device)
    return build_cmm_model(
        model=base_for_mapping,
        scope=cmm_args.get("mapping_scope", "decoder_only"),
        tile_shape=tuple(cmm_args.get("tile_shape", (128, 128))),
        rmin=float(cmm_args.get("rmin", 1e3)),
        rmax=float(cmm_args.get("rmax", 1e5)),
        cell_bits=int(cmm_args.get("cell_bits", 0)),
        # checkpoint 已保存写入后的 r_pos/r_neg，加载结构时不再重新注入写入噪声
        write_noise_std=0.0,
        read_noise_std=float(cmm_args.get("read_noise_std", 0.0)),
        adc_resolution=int(cmm_args.get("adc_resolution", 0)),
        dac_resolution=int(cmm_args.get("dac_resolution", 0)),
    )


def main() -> None:
    """加载数字模型，执行论文式 CMM 映射，并保存 CMM checkpoint。"""
    args = parse_args()
    device = torch.device(args.device)

    # 加载数字基线模型并冻结编码器
    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state_dict)
    freeze_encoder(model)
    model.to(device)
    model.eval()

    # 打印映射参数摘要
    linear_count = count_linear_layers_by_scope(model, args.scope)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Scope: {args.scope}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    print(f"Rmin/Rmax: {args.rmin}/{args.rmax}")
    print(f"Cell bits: {args.cell_bits}")
    print(f"Write/read noise std: {args.write_noise_std}/{args.read_noise_std}")
    print(f"ADC/DAC resolution: {args.adc_resolution}/{args.dac_resolution}")
    print(f"Decoder linear layers to map: {linear_count}")

    # 执行 CMM 映射
    cmm_model = build_cmm_model(
        model=model,
        scope=args.scope,
        tile_shape=(args.tile_rows, args.tile_cols),
        rmin=args.rmin,
        rmax=args.rmax,
        cell_bits=args.cell_bits,
        write_noise_std=args.write_noise_std,
        read_noise_std=args.read_noise_std,
        adc_resolution=args.adc_resolution,
        dac_resolution=args.dac_resolution,
    )
    cmm_model.to(device)
    cmm_model.eval()
    print(f"CMMLinear layers mapped: {count_cmm_linear_layers(cmm_model)}")

    # 保存 CMM checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cmm_v1",
            "original_checkpoint": str(args.checkpoint),
            "model_config": payload.get("model_config"),
            "vocab_stoi": payload.get("vocab_stoi"),
            "cmm_model_state_dict": cmm_model.state_dict(),
            "cmm_args": make_cmm_args(args),
            "num_cmm_linear": count_cmm_linear_layers(cmm_model),
        },
        args.output,
    )

    print(f"CMM model saved to: {args.output}")


if __name__ == "__main__":
    main()
