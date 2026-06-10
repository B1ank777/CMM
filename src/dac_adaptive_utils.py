from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from .evaluate_crosssim import build_val_loader, build_vocab_from_payload_or_data
from .map_crosssim import (
    build_model_from_crosssim_payload,
    build_model_from_payload,
    freeze_encoder,
    load_checkpoint,
    synchronize_crosssim_cores,
)


def is_mapped_crosssim_linear(name: str, module: nn.Module) -> bool:
    """判断是否为 decoder 范围内已经映射到 CrossSim 的 Linear 层。"""
    if not (name.startswith("layers.") or name == "output_proj"):
        return False
    return hasattr(module, "core") and hasattr(module.core, "core") and hasattr(module.core.core, "cores")


def mapped_crosssim_linears(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """返回 21 个 mapped Linear 层；数量不符时直接报错，避免实验统计错层。"""
    layers = [(name, module) for name, module in model.named_modules() if is_mapped_crosssim_linear(name, module)]
    if len(layers) != 21:
        raise AssertionError(f"Expected 21 mapped CrossSim Linear layers, found {len(layers)}.")
    return layers


def layer_group(layer_name: str) -> str:
    """把层名归到论文表格里常用的模块组。"""
    if layer_name == "output_proj":
        return "output_proj"
    if ".self_attn." in layer_name:
        return "self_attn"
    if ".cross_attn." in layer_name:
        return "cross_attn"
    if ".ffn." in layer_name:
        return "ffn"
    return "other"


def load_baseline_and_crosssim(
    baseline_checkpoint: Path,
    crosssim_checkpoint: Path,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, Dict[str, Any], Dict[str, Any]]:
    """加载数字基线和 CMM-CrossSim/CrossSim checkpoint，供三个 DAC 实验复用。"""
    base_payload, state = load_checkpoint(baseline_checkpoint, map_location="cpu")
    base_model = build_model_from_payload(base_payload)
    base_model.load_state_dict(state)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()

    crosssim_payload = torch.load(crosssim_checkpoint, map_location="cpu")
    crosssim_args = dict(crosssim_payload.get("crosssim_args") or {})
    if not crosssim_args:
        raise ValueError("CrossSim checkpoint missing 'crosssim_args'.")
    if "crosssim_model_state_dict" not in crosssim_payload:
        raise ValueError("CrossSim checkpoint missing 'crosssim_model_state_dict'.")

    # 中文注释：checkpoint 可能是在 GPU 后端保存的；分析脚本允许用 CPU smoke test 重建同一权重。
    crosssim_args["use_gpu"] = bool(device.type == "cuda")
    crosssim_model = build_model_from_crosssim_payload(base_model, crosssim_args, device)
    crosssim_model.load_state_dict(crosssim_payload["crosssim_model_state_dict"])
    synchronize_crosssim_cores(crosssim_model)
    crosssim_model.to(device)
    crosssim_model.eval()
    return base_model, crosssim_model, base_payload, crosssim_payload


def build_validation_loader_from_payload(
    payload: Dict[str, Any],
    coco_root: Path,
    batch_size: int,
    num_workers: int,
    subset_size: int,
):
    """按训练 checkpoint 的词表和 max_len 构建 COCO val loader。"""
    vocab = build_vocab_from_payload_or_data(payload, coco_root)
    max_len = int(payload["model_config"]["max_len"])
    loader = build_val_loader(
        coco_root=coco_root,
        vocab=vocab,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size,
    )
    return loader, vocab


@torch.no_grad()
def run_teacher_forcing_batches(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> int:
    """运行 teacher-forcing forward，用于触发 activation hooks。"""
    model.eval()
    processed = 0
    for batch_idx, (images, captions) in enumerate(loader, start=1):
        if batch_idx > max_batches:
            break
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        _ = model(images, captions[:, :-1])
        processed += int(images.shape[0])
    return processed


def quantize_symmetric(values: torch.Tensor, bits: int) -> torch.Tensor:
    """模拟 signed DAC：先裁剪到 [-1, 1]，再做对称均匀量化。"""
    clipped = values.clamp(-1.0, 1.0)
    if bits <= 0:
        return clipped
    levels = (1 << bits) - 1
    q = torch.round((clipped + 1.0) * (levels / 2.0)) / (levels / 2.0) - 1.0
    return q.clamp(-1.0, 1.0)


def _last_dim_bias(module: nn.Module, output: torch.Tensor) -> torch.Tensor | None:
    """读取数字 bias，并 reshape 到可与任意 batch/sequence 输出广播的形状。"""
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    shape = [1] * output.dim()
    shape[-1] = int(bias.numel())
    return bias.view(*shape)


def patch_adaptive_dac(
    model: nn.Module,
    mode: str,
    layer_scales: Dict[str, float] | None = None,
    eps: float = 1e-12,
):
    """给 mapped Linear 打补丁，比较 layer-wise / batch-wise adaptive DAC。

    中文说明：
    - CrossSim 当前 DAC 默认按 [-1, 1] 解释输入，超范围值会被裁剪。
    - adaptive DAC 在进入 AnalogLinear 前把输入除以 scale，使 DAC 看到的值落回 [-1, 1]。
    - 线性输出再乘回 scale；若 bias 保留在数字域，则先扣除 bias 再加回，避免 bias 被缩放。
    """
    if mode not in {"layer-wise", "batch-wise"}:
        raise ValueError("mode must be 'layer-wise' or 'batch-wise'.")
    if mode == "layer-wise" and not layer_scales:
        raise ValueError("layer-wise adaptive DAC requires non-empty layer_scales.")

    handles = []
    for layer_name, module in mapped_crosssim_linears(model):
        original_forward = module.forward

        def patched_forward(*inputs, _layer_name=layer_name, _module=module, _forward=original_forward, **kwargs):
            x = inputs[0]
            if mode == "layer-wise":
                scale_value = float(layer_scales[_layer_name])  # type: ignore[index]
                scale = torch.as_tensor(max(scale_value, eps), dtype=x.dtype, device=x.device)
            else:
                scale = x.detach().abs().amax().clamp_min(eps).to(dtype=x.dtype)

            x_scaled = x / scale
            output = _forward(x_scaled, *inputs[1:], **kwargs)
            bias = _last_dim_bias(_module, output)
            if bias is None:
                return output * scale
            return (output - bias) * scale + bias

        module.forward = patched_forward  # type: ignore[method-assign]
        handles.append((module, original_forward))

    return handles


def restore_patched_forwards(handles: Iterable[Tuple[nn.Module, Any]]) -> None:
    """恢复被 patch_adaptive_dac 临时替换的 forward。"""
    for module, original_forward in handles:
        module.forward = original_forward  # type: ignore[method-assign]
