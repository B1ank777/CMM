from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 运行示例：
#   :: 从 nn.Linear 创建无噪声 CMMLinear，验证前向等价
#   python -c "import torch; from src.cmm import CMMLinear; \
#       lin = torch.nn.Linear(8, 4); cmm = CMMLinear.from_linear(lin); \
#       x = torch.randn(2, 8); print((lin(x) - cmm(x)).abs().max())"
#
#   :: 递归替换 nn.Module 中所有 Linear 层
#   python -c "from src.cmm import convert_module_to_cmm; \
#       import torch; m = convert_module_to_cmm(torch.nn.Linear(8, 4), write_noise_std=0.001)"


class CMMLinear(nn.Module):
    """论文式 CMM 等效线性层。

    CMM 使用忆阻值而不是电导存权重。这里保存正/负两套阵列的内部状态 r，
    forward 时按 Rmem = Ron * r + Roff * (1-r) 还原等效权重幅值。

    r_pos/r_neg 表示归一化忆阻状态（1=最低阻=Ron, 0=最高阻=Roff），
    对应论文映射 w = 1 - r。
    weight_scale 存储原始 nn.Linear 权重的最大绝对值，用于还原幅值。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rmin: float = 1e3,
        rmax: float = 1e5,
        cell_bits: int = 0,
        read_noise_std: float = 0.0,
        tile_rows: int = 128,
        tile_cols: int = 128,
        adc_resolution: int = 0,
        dac_resolution: int = 0,
    ) -> None:
        super().__init__()
        if rmin <= 0 or rmax <= 0 or rmin >= rmax:
            raise ValueError("rmin/rmax 必须为正数，且 rmin < rmax。")
        if cell_bits < 0:
            raise ValueError("cell_bits 必须大于等于 0。")
        if read_noise_std < 0:
            raise ValueError("read_noise_std 必须大于等于 0。")
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError("tile_rows/tile_cols 必须为正整数。")
        if adc_resolution < 0 or dac_resolution < 0:
            raise ValueError("adc_resolution/dac_resolution 必须大于等于 0。")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rmin = float(rmin)
        self.rmax = float(rmax)
        self.cell_bits = int(cell_bits)
        self.read_noise_std = float(read_noise_std)
        self.tile_rows = int(tile_rows)
        self.tile_cols = int(tile_cols)
        self.adc_resolution = int(adc_resolution)
        self.dac_resolution = int(dac_resolution)

        # r_pos/r_neg：核心内部态，保存为 buffer（不参与 autograd）
        self.register_buffer("r_pos", torch.ones(out_features, in_features))
        self.register_buffer("r_neg", torch.ones(out_features, in_features))
        self.register_buffer("weight_scale", torch.ones(()))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rmin: float = 1e3,
        rmax: float = 1e5,
        cell_bits: int = 0,
        write_noise_std: float = 0.0,
        read_noise_std: float = 0.0,
        tile_rows: int = 128,
        tile_cols: int = 128,
        adc_resolution: int = 0,
        dac_resolution: int = 0,
    ) -> "CMMLinear":
        """从 nn.Linear 生成 CMMLinear，bias 第一版保留在数字域。"""
        if write_noise_std < 0:
            raise ValueError("write_noise_std 必须大于等于 0。")

        cmm = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rmin=rmin,
            rmax=rmax,
            cell_bits=cell_bits,
            read_noise_std=read_noise_std,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            adc_resolution=adc_resolution,
            dac_resolution=dac_resolution,
        )
        # 提取权重并按最大绝对值归一化到 [0, 1]
        weight = linear.weight.detach().to(dtype=torch.float32)
        scale = weight.abs().max()
        if float(scale) == 0.0:
            scale = torch.ones_like(scale)

        # 正权重 → r_pos 通道，负权重 → r_neg 通道，分别归一化到 0~1
        pos_norm = torch.clamp(weight, min=0.0) / scale
        neg_norm = torch.clamp(-weight, min=0.0) / scale
        pos_norm = cls._quantize_unit_interval(pos_norm, cell_bits)
        neg_norm = cls._quantize_unit_interval(neg_norm, cell_bits)

        # 论文映射 w = 1 - r；写入噪声作用在内部状态 r 上
        r_pos = 1.0 - pos_norm
        r_neg = 1.0 - neg_norm
        if write_noise_std > 0:
            r_pos = r_pos + torch.randn_like(r_pos) * write_noise_std
            r_neg = r_neg + torch.randn_like(r_neg) * write_noise_std
        # 裁剪到物理合法范围 [0, 1]
        cmm.r_pos.copy_(torch.clamp(r_pos, 0.0, 1.0))
        cmm.r_neg.copy_(torch.clamp(r_neg, 0.0, 1.0))
        cmm.weight_scale.copy_(scale)
        if linear.bias is not None:
            cmm.bias.data.copy_(linear.bias.detach())
        return cmm.to(device=linear.weight.device, dtype=linear.weight.dtype)

    @staticmethod
    def _quantize_unit_interval(values: torch.Tensor, cell_bits: int) -> torch.Tensor:
        """按 cell_bits 将 [0, 1] 状态量化；0 表示连续状态。

        量化公式：round(v * (2^bits - 1)) / (2^bits - 1)，均匀量化为 levels 级。
        """
        if cell_bits <= 0:
            return values
        levels = (1 << cell_bits) - 1
        return torch.round(values * levels) / levels

    def _memristance_to_weight(self, r_state: torch.Tensor) -> torch.Tensor:
        """由内部状态 r 计算等效归一化权重幅值。

        Rmem = Ron * r + Roff * (1 - r)，然后将 Rmem 线性映射回 [0, 1]。
        """
        # 计算忆阻值：r=1→Ron(rmin), r=0→Roff(rmax)
        rmem = self.rmin * r_state + self.rmax * (1.0 - r_state)
        if self.read_noise_std > 0:
            # 读噪声按忆阻值比例扰动，随后裁剪回物理阻值范围
            rmem = rmem + torch.randn_like(rmem) * self.read_noise_std * rmem.abs()
            rmem = torch.clamp(rmem, self.rmin, self.rmax)
        # 线性映射回 [0, 1]：(Rmem - Ron) / (Roff - Ron)
        return torch.clamp((rmem - self.rmin) / (self.rmax - self.rmin), 0.0, 1.0)

    def effective_weight(self) -> torch.Tensor:
        """返回当前 CMM 状态对应的数字等效权重矩阵。

        正通道减去负通道，再乘以 weight_scale 还原原始幅值。
        等价于 w_eff = (r_pos_weight - r_neg_weight) * scale。
        """
        pos = self._memristance_to_weight(self.r_pos)
        neg = self._memristance_to_weight(self.r_neg)
        return (pos - neg) * self.weight_scale

    @staticmethod
    def _quantize_symmetric(values: torch.Tensor, bits: int) -> torch.Tensor:
        """对称 signed uniform 量化；bits=0 表示理想通路，不做量化。

        第一版使用当前张量的动态最大绝对值作为量化范围，避免引入校准流程。
        """
        if bits <= 0:
            return values
        max_abs = values.detach().abs().amax()
        if float(max_abs) == 0.0:
            return values
        levels = (1 << (bits - 1)) - 1
        if levels <= 0:
            # 1 bit signed 情况只有符号位，保守退化为二值幅值。
            return torch.sign(values) * max_abs
        scaled = torch.clamp(values / max_abs, -1.0, 1.0)
        return torch.round(scaled * levels) / levels * max_abs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """按 CMM tile 分块执行线性变换，支持 [B, D] 与 [B, T, D] 等输入形状。

        先将有效权重按 tile_rows × tile_cols 切块，逐块计算 partial sum
        再拼回完整输出。DAC 量化作用在每个输入 tile，ADC 量化作用在每个
        partial sum；两者 resolution=0 时保持理想通路，数值等价 nn.Linear。
        """
        weight = self.effective_weight()

        # 按输出行分块（对应不同阵列的输出神经元组）
        outputs = []
        for out_start in range(0, self.out_features, self.tile_rows):
            out_end = min(out_start + self.tile_rows, self.out_features)
            y_block = None

            # 按输入列分块（对应阵列的输入维度切块）
            for in_start in range(0, self.in_features, self.tile_cols):
                in_end = min(in_start + self.tile_cols, self.in_features)
                x_tile = input[..., in_start:in_end]
                x_tile = self._quantize_symmetric(x_tile, self.dac_resolution)
                w_tile = weight[out_start:out_end, in_start:in_end]
                partial = F.linear(x_tile, w_tile, None)
                partial = self._quantize_symmetric(partial, self.adc_resolution)
                # 累加同一输出块对应不同输入块的 partial sum
                y_block = partial if y_block is None else y_block + partial

            outputs.append(y_block)

        # 拼接所有输出块
        output = torch.cat(outputs, dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output


def convert_module_to_cmm(
    module: nn.Module,
    *,
    rmin: float = 1e3,
    rmax: float = 1e5,
    cell_bits: int = 0,
    write_noise_std: float = 0.0,
    read_noise_std: float = 0.0,
    tile_rows: int = 128,
    tile_cols: int = 128,
    adc_resolution: int = 0,
    dac_resolution: int = 0,
) -> nn.Module:
    """递归替换模块中的 nn.Linear → CMMLinear，保持原模块结构不变。

    不修改 encoder 之外的模块拓扑；若 module 本身是 nn.Linear 则直接返回 CMMLinear。
    对含子模块的容器则 deepcopy 后递归处理每个 child。
    """
    # 直接是 Linear → 替换为 CMMLinear
    if isinstance(module, nn.Linear):
        return CMMLinear.from_linear(
            module,
            rmin=rmin,
            rmax=rmax,
            cell_bits=cell_bits,
            write_noise_std=write_noise_std,
            read_noise_std=read_noise_std,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            adc_resolution=adc_resolution,
            dac_resolution=dac_resolution,
        )

    # 容器模块 → 深拷贝后递归替换子模块
    converted = copy.deepcopy(module)
    for name, child in list(converted.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                converted,
                name,
                CMMLinear.from_linear(
                    child,
                    rmin=rmin,
                    rmax=rmax,
                    cell_bits=cell_bits,
                    write_noise_std=write_noise_std,
                    read_noise_std=read_noise_std,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                    adc_resolution=adc_resolution,
                    dac_resolution=dac_resolution,
                ),
            )
        else:
            setattr(
                converted,
                name,
                convert_module_to_cmm(
                    child,
                    rmin=rmin,
                    rmax=rmax,
                    cell_bits=cell_bits,
                    write_noise_std=write_noise_std,
                    read_noise_std=read_noise_std,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                    adc_resolution=adc_resolution,
                    dac_resolution=dac_resolution,
                ),
            )
    return converted


def count_cmm_linear_layers(module: nn.Module) -> int:
    """统计模型中的 CMMLinear 层数量，用于验证映射是否覆盖了预期层。"""
    return sum(1 for child in module.modules() if isinstance(child, CMMLinear))
