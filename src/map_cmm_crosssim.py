from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from .map_crosssim import (
    build_crosssim_model,
    build_model_from_payload,
    count_linear_layers_by_scope,
    freeze_encoder,
    load_checkpoint,
    make_crosssim_args,
    should_use_crosssim_gpu,
)

# 运行示例：
#   # 构建理想 CMM-on-CrossSim 基线（decoder layers + output_proj）
#   python -m src.map_cmm_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt
#
#   # 使用 CrossSim 后端加入 CMM 语义下的写入/读取噪声
#   python -m src.map_cmm_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --write-noise-std 0.001 --read-noise-std 0.0001 \
#       --output checkpoints/cmm_crosssim_conditions/caption_transformer_cmm_crosssim_it7.pt
#
#   # GPU 加速并指定 ADC/DAC 分辨率
#   python -m src.map_cmm_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --device cuda --use-gpu --adc-resolution 8 --dac-resolution 8


def parse_args() -> argparse.Namespace:
    """解析 CMM-on-CrossSim 映射参数。"""
    parser = argparse.ArgumentParser(
        description="使用 CrossSim AnalogLinear 正式承载 CMM decoder 线性层映射"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="原始训练检查点路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/cmm_crosssim_conditions/caption_transformer_cmm_crosssim_ideal.pt"),
        help="CMM-on-CrossSim checkpoint 输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="映射与推理设备；cuda 会启用 CrossSim GPU 后端",
    )
    parser.add_argument("--use-gpu", action="store_true", help="显式要求 CrossSim 使用 GPU")
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="映射范围：output_only=仅 LM head；layers_only=仅 decoder 层；decoder_only=全部 decoder 线性层",
    )
    parser.add_argument("--tile-rows", type=int, default=128, help="CMM 阵列最大行数")
    parser.add_argument("--tile-cols", type=int, default=128, help="CMM 阵列最大列数")
    parser.add_argument("--rmin", type=float, default=1e3, help="Ron，器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="Roff，器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--adc-resolution", type=int, default=0, help="ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=0, help="DAC 分辨率；0 表示理想 DAC")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="读噪声强度；0 表示关闭")
    parser.add_argument(
        "--write-noise-std",
        type=float,
        default=0.0,
        help="CMM 语义下的写入噪声；CrossSim 后端映射为 programming_error_std",
    )
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数；0 表示数字 bias")
    return parser.parse_args()


def make_cmm_crosssim_args(args: argparse.Namespace, use_gpu: bool) -> Dict[str, Any]:
    """整理 CMM 语义元数据，明确 write_noise_std 与 CrossSim programming_error 的对应关系。"""
    return {
        "tile_shape": (args.tile_rows, args.tile_cols),
        "mapping_scope": args.scope,
        "rmin": args.rmin,
        "rmax": args.rmax,
        "cell_bits": args.cell_bits,
        "adc_resolution": args.adc_resolution,
        "dac_resolution": args.dac_resolution,
        "read_noise_std": args.read_noise_std,
        "write_noise_std": args.write_noise_std,
        "crosssim_programming_error_std": args.write_noise_std,
        "bias_rows": args.bias_rows,
        "use_gpu": use_gpu,
    }


def build_cmm_crosssim_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
    use_gpu: bool,
) -> torch.nn.Module:
    """按 CMM 参数语义构建 CrossSim AnalogLinear 模型。"""
    # CrossSim 没有直接使用 CMM r-state；这里将 CMM 写入噪声语义落到 programming_error。
    return build_crosssim_model(
        model=model,
        scope=args.scope,
        tile_shape=(args.tile_rows, args.tile_cols),
        adc_resolution=args.adc_resolution,
        dac_resolution=args.dac_resolution,
        bias_rows=args.bias_rows,
        use_gpu=use_gpu,
        rmin=args.rmin,
        rmax=args.rmax,
        cell_bits=args.cell_bits,
        read_noise_std=args.read_noise_std,
        programming_error_std=args.write_noise_std,
    )


def main() -> None:
    """加载数字模型，执行 CMM-on-CrossSim 映射，并保存正式实验 checkpoint。"""
    args = parse_args()
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state_dict)
    freeze_encoder(model)
    model.to(device)
    model.eval()

    num_mapped_linear = count_linear_layers_by_scope(model, args.scope)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"CMM-on-CrossSim scope: {args.scope}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    print(f"Rmin/Rmax: {args.rmin}/{args.rmax}")
    print(f"Cell bits: {args.cell_bits}")
    print(f"Write/read noise std: {args.write_noise_std}/{args.read_noise_std}")
    print(f"ADC/DAC resolution: {args.adc_resolution}/{args.dac_resolution}")
    print(f"CrossSim GPU: {use_gpu}")
    print(f"Decoder linear layers to map: {num_mapped_linear}")

    crosssim_model = build_cmm_crosssim_model(model, args, use_gpu)
    crosssim_model.to(device)
    crosssim_model.eval()

    args.programming_error_std = args.write_noise_std
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cmm_crosssim_v1",
            "original_checkpoint": str(args.checkpoint),
            "model_config": payload.get("model_config"),
            "vocab_stoi": payload.get("vocab_stoi"),
            "crosssim_model_state_dict": crosssim_model.state_dict(),
            "crosssim_args": make_crosssim_args(args, use_gpu),
            "cmm_crosssim_args": make_cmm_crosssim_args(args, use_gpu),
            "num_mapped_linear": num_mapped_linear,
        },
        args.output,
    )

    print(f"CMM-on-CrossSim model saved to: {args.output}")


if __name__ == "__main__":
    main()
