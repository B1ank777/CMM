"""
将训练好的 CaptionTransformer 的 nn.Linear 层映射为忆阻器（MemTorch）等效模型。

功能:
    1. 从训练检查点加载原始模型
    2. 使用 memtorch 将所有权重矩阵替换为忆阻器交叉阵列（crossbar）仿真
    3. 保存映射后的忆阻器模型，用于推理阶段的硬件感知评估

原理:
    深度学习模型的 Linear 层本质是矩阵-向量乘法 (y = Wx + b)。
    忆阻器交叉阵列可以在模拟域中直接执行这一运算 —— 每个忆阻器的电导值
    对应权重矩阵的一个元素，输入电压对应激活值，输出电流经 ADC 转换后
    即为计算结果。

运行方式 (bash / Git Bash):
    不使用 C++ binding（纯 Python 仿真，速度慢但无需编译）:
        python -m src.map_memtorch --checkpoint checkpoints/caption_transformer_epoch_10.pt --output checkpoints/caption_transformer_memtorch.pt

    使用 C++ binding（加速仿真，需先编译 memtorch 的 C++ 扩展）:
        python -m src.map_memtorch --checkpoint checkpoints/caption_transformer_epoch_10.pt --output checkpoints/caption_transformer_memtorch_bindings.pt --use-bindings

    注意：不要在 PowerShell 中使用 \\ 换行，PowerShell 无法解析 -- 和 \\。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .models import CaptionTransformer


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将 CaptionTransformer 的 Linear 层映射为 MemTorch 忆阻器模型"
    )

    # --- 文件路径 ---
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="训练产出的 .pt 检查点路径",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("checkpoints/caption_transformer_memtorch.pt"),
        help="映射后的忆阻器模型保存路径",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )

    # --- MemTorch 映射选项 ---
    parser.add_argument(
        "--use-bindings", action="store_true",
        help="是否使用 memtorch 的 C++ binding 加速（需预先编译）",
    )

    # --- 交叉阵列（Crossbar）参数 ---
    parser.add_argument(
        "--tile-rows", type=int, default=128,
        help="交叉阵列分块的行数（权重矩阵太大时需分块映射到多个小阵列）",
    )
    parser.add_argument(
        "--tile-cols", type=int, default=128,
        help="交叉阵列分块的列数",
    )

    # --- 模拟电路参数 ---
    parser.add_argument(
        "--max-input-voltage", type=float, default=0.3,
        help="最大输入电压 (V)，用于将激活值线性缩放到安全电压范围",
    )
    parser.add_argument(
        "--adc-resolution", type=int, default=8,
        help="ADC 分辨率 (bit)，决定输出电流的量化精度",
    )

    # --- 忆阻器器件参数 ---
    parser.add_argument(
        "--ron", type=float, default=1e2,
        help="忆阻器低阻态 (ON) 电阻值，单位 Ω",
    )
    parser.add_argument(
        "--roff", type=float, default=1e4,
        help="忆阻器高阻态 (OFF) 电阻值，单位 Ω。Ron/Roff 比决定可区分的电导状态数",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 检查点加载
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: Path, map_location: str
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """加载训练检查点，返回 (元数据字典, 模型权重字典)。

    兼容两种格式:
        - 训练脚本保存的完整格式（含 model_state_dict、model_config 等顶层键）
        - 纯 state_dict 格式（直接就是权重字典）
    """
    payload = torch.load(path, map_location=map_location)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        # 完整训练检查点格式
        return payload, payload["model_state_dict"]

    if isinstance(payload, dict):
        # 裸 state_dict 格式，包装为统一结构
        return {"model_state_dict": payload}, payload

    raise ValueError("Unsupported checkpoint format.")


# ---------------------------------------------------------------------------
# 模型重建
# ---------------------------------------------------------------------------

def build_model_from_payload(payload: Dict[str, Any]) -> CaptionTransformer:
    """根据检查点中保存的 model_config 重建 CaptionTransformer 模型结构。

    注意:
        - 编码器使用 ResNet（与训练时一致）
        - pretrained_encoder=False：不从 torchvision 下载预训练权重，
          因为权重会从检查点中加载
    """
    cfg = payload.get("model_config")
    if cfg is None:
        raise ValueError(
            "Checkpoint missing model_config. "
            "Please use training checkpoint payload format."
        )

    model = CaptionTransformer(
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
    return model


def freeze_encoder(model: nn.Module) -> None:
    """冻结 CNN 编码器参数 —— 推理时不需要梯度。"""
    for p in model.encoder.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------------
# 忆阻器映射核心
# ---------------------------------------------------------------------------

def build_memristive_model(
    model: nn.Module,
    use_bindings: bool,
    tile_shape: Tuple[int, int],
    max_input_voltage: float,
    adc_resolution: int,
    ron: float,
    roff: float,
) -> nn.Module:
    """将模型中的所有 nn.Linear 层替换为忆阻器交叉阵列。

    工作流程:
        1. deepcopy 原始模型（避免修改原模型）
        2. patch_model 自动遍历所有 nn.Linear 子模块，将权重矩阵映射为
           忆阻器交叉阵列的等效电导矩阵
        3. 前向传播时，交叉阵列在模拟域计算矩阵-向量乘法，
           输出经 ADC 量化后返回数字域

    参数:
        model: 原始 CaptionTransformer 模型
        use_bindings: 是否使用 C++ binding 加速仿真
        tile_shape: 交叉阵列分块形状 (rows, cols)
        max_input_voltage: 最大输入电压
        adc_resolution: ADC 量化分辨率
        ron: ON 态电阻
        roff: OFF 态电阻

    返回:
        所有 Linear 层已被替换为忆阻器交叉阵列的模型副本

    容错:
        memtorch 不同版本的 VTEAM 参数接口可能不同。
        若默认参数不兼容，自动回退到空参数 {} 重试。
    """
    import memtorch
    from memtorch.bh.crossbar.Program import naive_program
    from memtorch.map.Input import naive_scale
    from memtorch.map.Parameter import naive_map
    from memtorch.mn.Module import patch_model

    # 第一次尝试：传入器件参数
    try:
        mem_model = patch_model(
            copy.deepcopy(model),                     # 深拷贝，不影响原始模型
            memristor_model=memtorch.bh.memristor.VTEAM,  # VTEAM 忆阻器模型
            memristor_model_params={"r_on": ron, "r_off": roff},
            module_parameters_to_patch=[nn.Linear],   # 仅替换 Linear 层
            mapping_routine=naive_map,                # 权重 → 电导的映射策略
            transistor=True,                          # 启用 1T1R 结构（每单元串联晶体管）
            programming_routine=naive_program,        # 电导写入策略
            tile_shape=tile_shape,                    # 交叉阵列分块大小
            max_input_voltage=max_input_voltage,      # 输入电压范围
            scaling_routine=naive_scale,              # 激活值 → 电压的缩放策略
            ADC_resolution=adc_resolution,            # ADC 分辨率
            ADC_overflow_rate=0.0,                    # ADC 溢出率（0 表示无溢出）
            quant_method="linear",                    # 量化方法
            use_bindings=use_bindings,                # 是否用 C++ binding
        )
    except TypeError:
        # 回退：某些 memtorch 版本 VTEAM 不接受 r_on/r_off 参数
        print("Warning: VTEAM with params failed, falling back to default params.")
        mem_model = patch_model(
            copy.deepcopy(model),
            memristor_model=memtorch.bh.memristor.VTEAM,
            memristor_model_params={},                # 使用默认器件参数
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

    return mem_model


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def count_linear_layers(module: nn.Module) -> int:
    """统计模型中 nn.Linear 层的数量（用于验证映射是否完整）。"""
    return sum(1 for m in module.modules() if isinstance(m, nn.Linear))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    """主入口：加载检查点 → 映射忆阻器 → 保存结果。"""
    args = parse_args()
    device = torch.device(args.device)

    # 1. 加载训练好的模型
    payload, state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state_dict)
    freeze_encoder(model)
    model.eval()

    linear_count = count_linear_layers(model)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Linear layers to map: {linear_count}")

    # 2. 将 Linear 层映射为忆阻器交叉阵列
    mem_model = build_memristive_model(
        model=model,
        use_bindings=args.use_bindings,
        tile_shape=(args.tile_rows, args.tile_cols),
        max_input_voltage=args.max_input_voltage,
        adc_resolution=args.adc_resolution,
        ron=args.ron,
        roff=args.roff,
    )
    mem_model.to(device)
    mem_model.eval()

    # 3. 保存忆阻器模型及映射参数
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
            },
        },
        args.output,
    )

    print(f"Memristive model saved to: {args.output}")


if __name__ == "__main__":
    main()
