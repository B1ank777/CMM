from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .models import CaptionTransformer

# 运行示例：
#   # 默认理想映射（无噪声、无量化），仅映射 decoder 线性层
#   python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt
#
#   # 指定 tile 尺寸与 ADC/DAC 分辨率
#   python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --tile-rows 64 --tile-cols 64 --adc-resolution 8 --dac-resolution 8
#
#   # 仅映射 LM head，添加读噪声与编程误差
#   python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --scope output_only --read-noise-std 0.01 --programming-error-std 0.005
#
#   # GPU 加速映射，自定义输出路径
#   python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
#       --device cuda --use-gpu --output checkpoints/crosssim_gpu.pt


def parse_args() -> argparse.Namespace:
    """解析 CrossSim 映射脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将 CaptionTransformer 的 decoder 线性层映射为 CrossSim 交叉阵列"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练检查点路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/caption_transformer_crosssim.pt"),
        help="CrossSim 映射后模型的输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="映射与推理设备；cuda 会启用 CrossSim GPU 后端",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="显式要求 CrossSim 使用 GPU；若 device 不是 cuda 将直接报错",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="映射范围：output_only=仅 LM head；layers_only=仅 decoder 层；decoder_only=全部 decoder 线性层",
    )
    parser.add_argument("--tile-rows", type=int, default=128, help="交叉阵列最大行数")
    parser.add_argument("--tile-cols", type=int, default=128, help="交叉阵列最大列数")
    parser.add_argument("--adc-resolution", type=int, default=0, help="ADC 分辨率；0 表示理想 ADC")
    parser.add_argument("--dac-resolution", type=int, default=0, help="DAC 分辨率；0 表示理想 DAC")
    parser.add_argument("--bias-rows", type=int, default=0, help="bias 映射到阵列时使用的额外行数；0 表示数字 bias")
    parser.add_argument("--rmin", type=float, default=1e3, help="器件最小电阻")
    parser.add_argument("--rmax", type=float, default=1e5, help="器件最大电阻")
    parser.add_argument("--cell-bits", type=int, default=0, help="单元量化 bit 数；0 表示连续电导")
    parser.add_argument("--read-noise-std", type=float, default=0.0, help="读噪声强度；0 表示关闭")
    parser.add_argument("--programming-error-std", type=float, default=0.0, help="写入误差强度；0 表示关闭")
    return parser.parse_args()


def load_checkpoint(
    path: Path, map_location: str
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """加载训练检查点，兼容 payload 与纯 state_dict 两种格式。"""
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload, payload["model_state_dict"]
    if isinstance(payload, dict):
        return {"model_state_dict": payload}, payload
    raise ValueError("Unsupported checkpoint format.")


def build_model_from_payload(payload: Dict[str, Any]) -> CaptionTransformer:
    """根据 checkpoint 内保存的 model_config 重建 CaptionTransformer。"""
    cfg = payload.get("model_config")
    if cfg is None:
        raise ValueError(
            "Checkpoint missing model_config. Please use training checkpoint payload format."
        )

    return CaptionTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
        encoder_type="resnet",
        pad_id=cfg.get("pad_id", 0),
        pretrained_encoder=False,
    )


def freeze_encoder(model: nn.Module) -> None:
    """冻结 CNN 编码器参数，保持 CrossSim 映射只作用于 decoder。"""
    for param in model.encoder.parameters():
        param.requires_grad = False


def should_use_crosssim_gpu(device: torch.device, requested_use_gpu: bool = False) -> bool:
    """根据目标设备决定 CrossSim 后端；显式 GPU 请求失败时不降级。"""
    use_gpu = requested_use_gpu or device.type == "cuda"
    if not use_gpu:
        return False
    if device.type != "cuda":
        raise RuntimeError("已请求 CrossSim GPU，但 --device 不是 cuda。请使用 --device cuda。")
    if not torch.cuda.is_available():
        raise RuntimeError("已请求 CrossSim GPU，但当前 PyTorch 无法使用 CUDA。")
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        raise RuntimeError("已请求 CrossSim GPU，但 cupy 不可用或 CUDA 初始化失败。") from exc
    return True


def build_crosssim_params(
    *,
    tile_shape: Tuple[int, int],
    adc_resolution: int = 0,
    dac_resolution: int = 0,
    use_gpu: bool = False,
    rmin: float = 1e3,
    rmax: float = 1e5,
    cell_bits: int = 0,
    read_noise_std: float = 0.0,
    programming_error_std: float = 0.0,
):
    """构建 CrossSimParameters，默认采用无量化、无噪声的理想基线。"""
    from simulator import CrossSimParameters

    params = CrossSimParameters()
    params.update(
        {
            "core.style": "BALANCED",
            "core.rows_max": int(tile_shape[0]),
            "core.cols_max": int(tile_shape[1]),
            "core.output_dtype": "FLOAT32",
            "simulation.useGPU": bool(use_gpu),
            "xbar.device.Rmin": float(rmin),
            "xbar.device.Rmax": float(rmax),
            "xbar.device.cell_bits": int(cell_bits),
            # CrossSim 3.2.0 的 ADC/DAC 是 mvm/vmm 成对参数，不能直接写 xbar.adc.bits。
            "xbar.adc.mvm.bits": int(adc_resolution),
            "xbar.adc.vmm.bits": int(adc_resolution),
            "xbar.dac.mvm.bits": int(dac_resolution),
            "xbar.dac.vmm.bits": int(dac_resolution),
            "xbar.device.read_noise.enable": read_noise_std > 0,
            "xbar.device.read_noise.model": "NormalProportionalDevice",
            "xbar.device.read_noise.magnitude": float(read_noise_std),
            "xbar.device.programming_error.enable": programming_error_std > 0,
            "xbar.device.programming_error.model": "NormalIndependentDevice",
            "xbar.device.programming_error.magnitude": float(programming_error_std),
            "xbar.device.drift_error.enable": False,
            "xbar.device.nonlinear_IV.enable": False,
        }
    )
    return params


def convert_module_to_crosssim(module: nn.Module, params, bias_rows: int) -> nn.Module:
    """将指定子模块中的 nn.Linear 转换为 CrossSim AnalogLinear。"""
    from simulator.algorithms.dnn.torch import from_torch

    return from_torch(module, params, bias_rows=bias_rows)


def build_crosssim_model(
    model: nn.Module,
    scope: str,
    tile_shape: Tuple[int, int],
    adc_resolution: int,
    dac_resolution: int,
    bias_rows: int,
    use_gpu: bool,
    rmin: float = 1e3,
    rmax: float = 1e5,
    cell_bits: int = 0,
    read_noise_std: float = 0.0,
    programming_error_std: float = 0.0,
) -> nn.Module:
    """按 scope 将 CaptionTransformer 的 decoder 线性层映射为 CrossSim 层。"""
    crosssim_model = copy.deepcopy(model)
    if not hasattr(crosssim_model, "layers") or not hasattr(crosssim_model, "output_proj"):
        raise ValueError("build_crosssim_model expects a CaptionTransformer-like model.")

    params = build_crosssim_params(
        tile_shape=tile_shape,
        adc_resolution=adc_resolution,
        dac_resolution=dac_resolution,
        use_gpu=use_gpu,
        rmin=rmin,
        rmax=rmax,
        cell_bits=cell_bits,
        read_noise_std=read_noise_std,
        programming_error_std=programming_error_std,
    )

    if scope in {"layers_only", "decoder_only"}:
        crosssim_model.layers = convert_module_to_crosssim(crosssim_model.layers, params, bias_rows)
    if scope in {"output_only", "decoder_only"}:
        crosssim_model.output_proj = convert_module_to_crosssim(crosssim_model.output_proj, params, bias_rows)
    return crosssim_model


def synchronize_crosssim_cores(model: nn.Module) -> None:
    """load_state_dict 会原地改权重；这里显式同步 CrossSim 内部阵列。"""
    from simulator.algorithms.dnn.torch import synchronize

    synchronize(model)


def count_linear_layers_by_scope(module: nn.Module, scope: str) -> int:
    """统计 decoder 中会被 CrossSim 映射的 nn.Linear 层数量。"""
    count = 0
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if name.startswith("encoder."):
            continue
        if scope == "output_only" and name != "output_proj":
            continue
        if scope == "layers_only" and not name.startswith("layers."):
            continue
        if scope == "decoder_only" and not (name.startswith("layers.") or name == "output_proj"):
            continue
        count += 1
    return count


def make_crosssim_args(args: argparse.Namespace, use_gpu: bool) -> Dict[str, Any]:
    """整理保存到 checkpoint 的 CrossSim 配置元数据。"""
    return {
        "tile_shape": (args.tile_rows, args.tile_cols),
        "adc_resolution": args.adc_resolution,
        "dac_resolution": args.dac_resolution,
        "bias_rows": args.bias_rows,
        "mapping_scope": args.scope,
        "use_gpu": use_gpu,
        "rmin": args.rmin,
        "rmax": args.rmax,
        "cell_bits": args.cell_bits,
        "read_noise_std": args.read_noise_std,
        "programming_error_std": args.programming_error_std,
    }


def build_model_from_crosssim_payload(
    baseline_model: nn.Module,
    crosssim_args: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """根据 CrossSim checkpoint 元数据重建 analog 模型结构。"""
    use_gpu = should_use_crosssim_gpu(device, bool(crosssim_args.get("use_gpu", False)))
    base_for_mapping = baseline_model.to(device)
    return build_crosssim_model(
        model=base_for_mapping,
        scope=crosssim_args.get("mapping_scope", "decoder_only"),
        tile_shape=tuple(crosssim_args.get("tile_shape", (128, 128))),
        adc_resolution=int(crosssim_args.get("adc_resolution", 0)),
        dac_resolution=int(crosssim_args.get("dac_resolution", 0)),
        bias_rows=int(crosssim_args.get("bias_rows", 0)),
        use_gpu=use_gpu,
        rmin=float(crosssim_args.get("rmin", 1e3)),
        rmax=float(crosssim_args.get("rmax", 1e5)),
        cell_bits=int(crosssim_args.get("cell_bits", 0)),
        read_noise_std=float(crosssim_args.get("read_noise_std", 0.0)),
        programming_error_std=float(crosssim_args.get("programming_error_std", 0.0)),
    )


def main() -> None:
    """加载数字模型，执行 CrossSim 映射，并保存 CrossSim checkpoint。"""
    args = parse_args()
    device = torch.device(args.device)
    use_gpu = should_use_crosssim_gpu(device, args.use_gpu)

    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state_dict)
    freeze_encoder(model)
    model.to(device)
    model.eval()

    linear_count = count_linear_layers_by_scope(model, args.scope)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Scope: {args.scope}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    print(f"ADC/DAC resolution: {args.adc_resolution}/{args.dac_resolution}")
    print(f"CrossSim GPU: {use_gpu}")
    print(f"Decoder linear layers to map: {linear_count}")

    crosssim_model = build_crosssim_model(
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
        programming_error_std=args.programming_error_std,
    )
    crosssim_model.to(device)
    crosssim_model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "original_checkpoint": str(args.checkpoint),
            "model_config": payload.get("model_config"),
            "vocab_stoi": payload.get("vocab_stoi"),
            "crosssim_model_state_dict": crosssim_model.state_dict(),
            "crosssim_args": make_crosssim_args(args, use_gpu),
        },
        args.output,
    )

    print(f"CrossSim model saved to: {args.output}")


if __name__ == "__main__":
    main()
