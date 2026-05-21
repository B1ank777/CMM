from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .models import CaptionTransformer

# ============================================================================
# 忆阻器交叉阵列映射脚本
# ============================================================================
# 功能:
#     将训练好的 CaptionTransformer 模型中指定的 nn.Linear 层替换为
#     MemTorch 仿真的忆阻器交叉阵列（VTEAM 模型），模拟模拟计算硬件的推理行为。
#
# 映射范围 (--scope):
#     output_only : 仅映射输出投影层 output_proj（LM head）
#     layers_only : 仅映射 DecoderLayer 内部的 Q/K/V/O/FFN1/FFN2 线性层
#     decoder_only: 映射所有 decoder 线性层（layers + output_proj）
#
# 使用方式:
#     python -m src.map_memtorch --checkpoint <训练检查点> --scope decoder_only
# ============================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="将 CaptionTransformer 的 decoder 线性层映射为 MemTorch 忆阻器交叉阵列"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练检查点路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/caption_transformer_memtorch.pt"),
        help="忆阻器映射后模型的输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    parser.add_argument("--use-bindings", action="store_true", help="启用 MemTorch C++ binding 加速")
    parser.add_argument(
        "--scope",
        type=str,
        default="decoder_only",
        choices=["output_only", "layers_only", "decoder_only"],
        help="映射范围：output_only=仅LM头 | layers_only=仅decoder层内线性层 | decoder_only=全部decoder线性层",
    )
    parser.add_argument("--tile-rows", type=int, default=128, help="交叉阵列分块行数")
    parser.add_argument("--tile-cols", type=int, default=128, help="交叉阵列分块列数")
    parser.add_argument("--max-input-voltage", type=float, default=0.3, help="最大输入电压 (V)")
    parser.add_argument("--adc-resolution", type=int, default=8, help="ADC 分辨率 (bit)")
    parser.add_argument("--ron", type=float, default=1e2, help="忆阻器低阻态电阻 (Ω)")
    parser.add_argument("--roff", type=float, default=1e4, help="忆阻器高阻态电阻 (Ω)")
    return parser.parse_args()


def load_checkpoint(
    path: Path, map_location: str
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """加载检查点文件，兼容标准训练格式与纯 state_dict 格式。

    返回:
        (payload_dict, state_dict) 元组，payload_dict 始终包含 "model_state_dict" 键。
    """
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload, payload["model_state_dict"]
    if isinstance(payload, dict):
        return {"model_state_dict": payload}, payload
    raise ValueError("Unsupported checkpoint format.")


def build_model_from_payload(payload: Dict[str, Any]) -> CaptionTransformer:
    """从检查点配置中重建 CaptionTransformer 模型结构（未加载权重）。

    需要检查点中包含 model_config 字段，否则无法确定模型超参数。
    """
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
    """冻结 CNN 编码器参数，推理时不计算梯度以节省显存。"""
    for p in model.encoder.parameters():
        p.requires_grad = False


def patch_memtorch_linear_input_shapes(module: nn.Module) -> None:
    """修补 MemTorch Linear 模块使其接受秩 > 2 的输入张量。

    MemTorch 的默认 Linear 仅支持 1D/2D 输入，但 Transformer decoder 中
    存在 (batch, seq, d_model) 的三维输入。此函数将每个 MemTorch Linear
    的 forward 替换为带 reshape 的版本：先将高维输入展平为 2D，前向后再恢复形状。

    同时处理 batch_size=1 时 MemTorch 不稳定的问题（通过临时复制一份输入）。
    """
    for child in module.modules():
        if type(child).__name__ != "Linear":
            continue
        if not type(child).__module__.startswith("memtorch"):
            continue
        if hasattr(child, "_original_memtorch_forward"):
            continue

        child._original_memtorch_forward = child.forward

        def _forward_with_reshape(self, input_tensor):
            def _forward_2d_stable(tensor_2d):
                # MemTorch Linear 在 batch_size=1 时不稳定，临时复制一份绕过
                if tensor_2d.shape[0] == 1:
                    duplicated = torch.cat([tensor_2d, tensor_2d], dim=0)
                    output = self._original_memtorch_forward(duplicated)
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    return output[:1]

                output = self._original_memtorch_forward(tensor_2d)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                return output

            if input_tensor.dim() == 1:
                output = _forward_2d_stable(input_tensor.unsqueeze(0))
                return output.squeeze(0)

            if input_tensor.dim() == 2:
                return _forward_2d_stable(input_tensor)

            input_shape = input_tensor.shape
            flat_input = input_tensor.reshape(-1, input_shape[-1])
            flat_output = _forward_2d_stable(flat_input)
            return flat_output.reshape(*input_shape[:-1], flat_output.shape[-1])

        child.forward = MethodType(_forward_with_reshape, child)


def patch_module_memristive(
    module: nn.Module,
    use_bindings: bool,
    tile_shape: Tuple[int, int],
    max_input_voltage: float,
    adc_resolution: int,
    ron: float,
    roff: float,
) -> nn.Module:
    """对指定模块执行 MemTorch 忆阻器映射。

    使用 VTEAM 忆阻器模型 + naive 映射/编程/缩放策略，
    将模块内所有 nn.Linear 层替换为忆阻器交叉阵列等效电路。
    映射后自动修补 forward 以支持高维输入。
    """
    import memtorch
    from memtorch.bh.crossbar.Program import naive_program
    from memtorch.map.Input import naive_scale
    from memtorch.map.Parameter import naive_map
    from memtorch.mn.Module import patch_model

    try:
        patched = patch_model(
            module,
            memristor_model=memtorch.bh.memristor.VTEAM,
            memristor_model_params={"r_on": ron, "r_off": roff},
            module_parameters_to_patch=[nn.Linear],
            mapping_routine=naive_map,
            transistor=True,
            programming_routine=naive_program,
            tile_shape=tile_shape,
            max_input_voltage=max_input_voltage,
            scaling_routine=naive_scale,
            ADC_resolution=adc_resolution,
            ADC_overflow_rate=0.0,
            quant_method="linear",
            use_bindings=use_bindings,
        )
    except TypeError:
        print("Warning: VTEAM with params failed, falling back to default params.")
        patched = patch_model(
            module,
            memristor_model=memtorch.bh.memristor.VTEAM,
            memristor_model_params={},
            module_parameters_to_patch=[nn.Linear],
            mapping_routine=naive_map,
            transistor=True,
            programming_routine=naive_program,
            tile_shape=tile_shape,
            max_input_voltage=max_input_voltage,
            scaling_routine=naive_scale,
            ADC_resolution=adc_resolution,
            ADC_overflow_rate=0.0,
            quant_method="linear",
            use_bindings=use_bindings,
        )

    patch_memtorch_linear_input_shapes(patched)
    return patched


def build_memristive_model(
    model: nn.Module,
    use_bindings: bool,
    scope: str,
    tile_shape: Tuple[int, int],
    max_input_voltage: float,
    adc_resolution: int,
    ron: float,
    roff: float,
) -> nn.Module:
    """按 scope 参数选择性映射 decoder 子模块到忆阻器交叉阵列。

    scope 取值:
        output_only : 仅映射 output_proj（词表投影层 / LM head）
        layers_only : 仅映射 layers（DecoderLayer 内的 Q/K/V/O/FFN1/FFN2）
        decoder_only: 映射以上两者（完整 decoder 忆阻器化）

    编码器（ResNet）始终保留为数字域，不参与映射。
    使用深拷贝避免修改原始模型。
    """
    mem_model = copy.deepcopy(model)
    if not hasattr(mem_model, "layers") or not hasattr(mem_model, "output_proj"):
        raise ValueError("build_memristive_model expects a CaptionTransformer-like model.")

    if scope in {"layers_only", "decoder_only"}:
        mem_model.layers = patch_module_memristive(
            mem_model.layers,
            use_bindings=use_bindings,
            tile_shape=tile_shape,
            max_input_voltage=max_input_voltage,
            adc_resolution=adc_resolution,
            ron=ron,
            roff=roff,
        )
    if scope in {"output_only", "decoder_only"}:
        mem_model.output_proj = patch_module_memristive(
            mem_model.output_proj,
            use_bindings=use_bindings,
            tile_shape=tile_shape,
            max_input_voltage=max_input_voltage,
            adc_resolution=adc_resolution,
            ron=ron,
            roff=roff,
        )
    return mem_model


def count_linear_layers_by_scope(module: nn.Module, scope: str) -> int:
    """统计解码器中将会被映射的 nn.Linear 层数量（仅用于调试输出）。"""
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


def main() -> None:
    """主流程：加载数字模型 → 选择性映射 → 保存忆阻器模型检查点。"""
    args = parse_args()
    device = torch.device(args.device)

    # 加载训练检查点，重建模型结构与权重
    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state_dict)
    freeze_encoder(model)  # 冻结编码器，节省推理显存
    model.eval()

    # 统计并输出映射范围信息
    linear_count = count_linear_layers_by_scope(model, args.scope)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Scope: {args.scope}")
    print(f"Tile shape: ({args.tile_rows}, {args.tile_cols})")
    print(f"ADC resolution: {args.adc_resolution}")
    print(f"Max input voltage: {args.max_input_voltage}")
    print(f"Decoder linear layers to map: {linear_count}")

    # 执行忆阻器交叉阵列映射
    mem_model = build_memristive_model(
        model=model,
        use_bindings=args.use_bindings,
        scope=args.scope,
        tile_shape=(args.tile_rows, args.tile_cols),
        max_input_voltage=args.max_input_voltage,
        adc_resolution=args.adc_resolution,
        ron=args.ron,
        roff=args.roff,
    )
    mem_model.to(device)
    mem_model.eval()

    # 保存映射后的模型，附带元数据便于后续追溯
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "original_checkpoint": str(args.checkpoint),
            "model_config": payload.get("model_config"),
            "vocab_stoi": payload.get("vocab_stoi"),
            "mem_model_state_dict": mem_model.state_dict(),
            "memtorch_args": {
                "use_bindings": args.use_bindings,
                "tile_shape": (args.tile_rows, args.tile_cols),
                "max_input_voltage": args.max_input_voltage,
                "adc_resolution": args.adc_resolution,
                "r_on": args.ron,
                "r_off": args.roff,
                "mapping_scope": args.scope,  # 记录映射范围，供评估脚本识别
            },
        },
        args.output,
    )

    print(f"Memristive model saved to: {args.output}")


if __name__ == "__main__":
    main()
