"""估计 CMM-CrossSim decoder peripheral-aware read energy。

运行示例:
    conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 1 --batch-size 1 --device cpu
    conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 16 --batch-size 1 --device cuda --use-gpu
    conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 1 --batch-size 1 --device cpu --skip-ngspice --adc-energy-pj-per-conv 1.0 --dac-energy-pj-per-conv 0.1 --digital-mac-energy-pj 0.9 --peripheral-energy-source "parameterized_literature_model"

说明:
    主能耗口径为 crossbar core read energy + ADC model + DAC model +
    optional digital accumulation / bias energy。数字 MAC 对照只覆盖同一批
    decoder mapped Linear，不包含 encoder、LayerNorm、softmax、residual。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_crossbar_read import run_matrix_experiment
from src.evaluate_crosssim import build_val_loader, build_vocab_from_payload_or_data
from src.map_crosssim import (
    build_crosssim_model,
    build_model_from_crosssim_payload,
    build_model_from_payload,
    freeze_encoder,
    load_checkpoint,
    synchronize_crosssim_cores,
)


ACTIVATION_SCALE = "per_vector"
ENERGY_SCOPE = "peripheral_aware_read_energy"
EXCLUDED = "sense,layernorm,softmax,residual,encoder"
DEFAULT_DIGITAL_MAC_ENERGY_PJ = 1.0
DEFAULT_BASELINE = Path("checkpoints/caption_transformer_epoch_10.pt")
DEFAULT_CMM_CROSSSIM = Path("checkpoints/caption_transformer_array-128x128_cmm_crosssim.pt")
DEFAULT_CROSSSIM_ONLY = Path("checkpoints/caption_transformer_crosssim_decoder.pt")
DEFAULT_OUTPUT_DIR = Path("experiments/spice/results/cmm_crosssim_read_energy")


@dataclass
class TileRecord:
    """保存单个 CrossSim core tile 的电导、电阻和能耗统计。"""

    tile_id: str
    layer_name: str
    group: str
    polarity: str
    core_row: int
    core_col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    g_raw: np.ndarray
    r_physical: np.ndarray
    conductance_by_input: np.ndarray
    tile_total_energy_j: float = 0.0
    tile_total_power_w: float = 0.0
    tile_total_read_current_a: float = 0.0
    vector_count: int = 0
    nonzero_vector_count: int = 0
    vector_powers_w: list[float] = field(default_factory=list)
    vector_energies_j: list[float] = field(default_factory=list)
    vector_voltages: list[np.ndarray] = field(default_factory=list)

    @property
    def tile_mean_vector_power_w(self) -> float:
        if not self.vector_powers_w:
            return 0.0
        return self.tile_total_power_w / len(self.vector_powers_w)

    @property
    def tile_mean_vector_read_current_a(self) -> float:
        if self.vector_count == 0:
            return 0.0
        return self.tile_total_read_current_a / self.vector_count


@dataclass
class DigitalMacRecord:
    """保存同一 mapped Linear 的数字 MAC 参考统计。"""

    layer_name: str
    group: str
    in_features: int
    out_features: int
    vector_count: int = 0

    @property
    def macs_per_vector(self) -> int:
        return self.in_features * self.out_features

    @property
    def total_macs(self) -> int:
        return self.vector_count * self.macs_per_vector


@dataclass
class PeripheralEnergyModel:
    """参数化外围能耗模型；单位统一为 pJ/op 或 pJ/conversion。"""

    adc_energy_pj_per_conv: float
    dac_energy_pj_per_conv: float
    digital_accum_energy_pj_per_op: float
    bias_energy_pj_per_op: float


@dataclass
class EnergyBreakdown:
    """保存 core 与外围能耗分项，便于 summary 和 tile 明细复用。"""

    core_read_energy_j: float = 0.0
    adc_energy_j: float = 0.0
    dac_energy_j: float = 0.0
    digital_accum_energy_j: float = 0.0
    bias_energy_j: float = 0.0
    adc_conversions: float = 0.0
    dac_conversions: float = 0.0
    digital_accum_ops: float = 0.0
    bias_ops: float = 0.0

    @property
    def total_energy_j(self) -> float:
        return (
            self.core_read_energy_j
            + self.adc_energy_j
            + self.dac_energy_j
            + self.digital_accum_energy_j
            + self.bias_energy_j
        )

    def add(self, other: "EnergyBreakdown") -> None:
        self.core_read_energy_j += other.core_read_energy_j
        self.adc_energy_j += other.adc_energy_j
        self.dac_energy_j += other.dac_energy_j
        self.digital_accum_energy_j += other.digital_accum_energy_j
        self.bias_energy_j += other.bias_energy_j
        self.adc_conversions += other.adc_conversions
        self.dac_conversions += other.dac_conversions
        self.digital_accum_ops += other.digital_accum_ops
        self.bias_ops += other.bias_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate CMM-CrossSim peripheral-aware read energy with ngspice core validation.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE, help="数字基线 checkpoint")
    parser.add_argument("--cmm-crosssim-checkpoint", type=Path, default=DEFAULT_CMM_CROSSSIM, help="CMM-CrossSim checkpoint")
    parser.add_argument("--crosssim-only-checkpoint", type=Path, default=DEFAULT_CROSSSIM_ONLY, help="CrossSim-only decoder checkpoint")
    parser.add_argument("--skip-crosssim-only", action="store_true", help="不计算普通 CrossSim-only 能耗对照")
    parser.add_argument("--coco-root", type=Path, default=None, help="COCO 根目录；默认复用项目 loader 默认值")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--use-gpu", action="store_true", help="让 CrossSim 使用 GPU 后端；需要 device=cuda")
    parser.add_argument("--limit", type=int, default=16, help="验证集样本数量")
    parser.add_argument("--batch-size", type=int, default=1, help="DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--vread", type=float, default=0.1, help="每个 activation vector 最大幅度缩放到的读电压/V")
    parser.add_argument("--pulse-ns", type=float, default=10.0, help="read pulse 宽度/ns")
    parser.add_argument("--spice-tran-step-ns", type=float, default=None, help="ngspice 验证 .tran step/ns；默认由 run_crossbar_read.py 决定")
    parser.add_argument("--spice-pulse-rise-ns", type=float, default=0.001, help="ngspice 验证 PULSE 上升沿/ns")
    parser.add_argument("--spice-pulse-fall-ns", type=float, default=0.001, help="ngspice 验证 PULSE 下降沿/ns")
    parser.add_argument("--ngspice", default="ngspice", help="ngspice 可执行文件路径")
    parser.add_argument("--skip-ngspice", action="store_true", help="只做解析能耗，不生成 high/median/low SPICE 验证")
    parser.add_argument(
        "--digital-mac-energy-pj",
        type=float,
        default=DEFAULT_DIGITAL_MAC_ENERGY_PJ,
        help="数字 MAC 能耗参考值，单位 pJ/MAC；默认仅作为可配置参考线",
    )
    parser.add_argument("--adc-energy-pj-per-conv", type=float, default=0.0, help="ADC 每次转换能耗，单位 pJ/conversion")
    parser.add_argument("--dac-energy-pj-per-conv", type=float, default=0.0, help="DAC 每次转换能耗，单位 pJ/conversion")
    parser.add_argument(
        "--digital-accum-energy-pj-per-op",
        type=float,
        default=0.0,
        help="tile partial sum 数字累加每次操作能耗，单位 pJ/op；0 表示不计入",
    )
    parser.add_argument(
        "--bias-energy-pj-per-op",
        type=float,
        default=0.0,
        help="bias 加法每次操作能耗，单位 pJ/op；0 表示不计入",
    )
    parser.add_argument(
        "--peripheral-energy-source",
        type=str,
        default="not_specified",
        help="ADC/DAC/digital/bias 参数来源说明，会写入 manifest 和 CSV",
    )
    return parser.parse_args()


def load_baseline_template(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """加载原始数字模型结构，作为 CrossSim / CMM-CrossSim 重建模板。"""
    payload, state_dict = load_checkpoint(checkpoint, map_location="cpu")
    base_model = build_model_from_payload(payload)
    base_model.load_state_dict(state_dict)
    freeze_encoder(base_model)
    base_model.to(device)
    base_model.eval()
    return base_model, payload


def load_cmm_crosssim_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """加载 baseline 结构并按 CMM-CrossSim checkpoint 元数据重建 AnalogLinear 模型。"""
    base_model, payload = load_baseline_template(args.baseline_checkpoint, device)

    crosssim_payload = torch.load(args.cmm_crosssim_checkpoint, map_location="cpu")
    if crosssim_payload.get("format") != "cmm_crosssim_v1":
        raise ValueError(f"Expected cmm_crosssim_v1 checkpoint, got {crosssim_payload.get('format')!r}.")
    cmm_args = crosssim_payload.get("cmm_crosssim_args")
    if not cmm_args:
        raise ValueError("CMM-CrossSim checkpoint missing cmm_crosssim_args.")
    if "crosssim_model_state_dict" not in crosssim_payload:
        raise ValueError("CMM-CrossSim checkpoint missing crosssim_model_state_dict.")

    use_gpu = bool(args.use_gpu or device.type == "cuda")
    if use_gpu and device.type != "cuda":
        raise RuntimeError("--use-gpu requires --device cuda.")

    model = build_crosssim_model(
        model=base_model,
        scope=cmm_args.get("mapping_scope", "decoder_only"),
        tile_shape=tuple(cmm_args.get("tile_shape", (128, 128))),
        adc_resolution=int(cmm_args.get("adc_resolution", 0)),
        dac_resolution=int(cmm_args.get("dac_resolution", 0)),
        bias_rows=int(cmm_args.get("bias_rows", 0)),
        use_gpu=use_gpu,
        rmin=float(cmm_args.get("rmin", 1e3)),
        rmax=float(cmm_args.get("rmax", 1e5)),
        cell_bits=int(cmm_args.get("cell_bits", 0)),
        read_noise_std=float(cmm_args.get("read_noise_std", 0.0)),
        programming_error_std=float(cmm_args.get("crosssim_programming_error_std", cmm_args.get("write_noise_std", 0.0))),
    )
    model.load_state_dict(crosssim_payload["crosssim_model_state_dict"])

    # load_state_dict 只写回 PyTorch 参数；必须同步到 CrossSim core 后才能提取真实 matrix。
    from simulator.algorithms.dnn.torch import synchronize

    synchronize(model)
    model.to(device)
    model.eval()
    return model, payload, crosssim_payload


def load_crosssim_only_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """加载普通 CrossSim-only decoder 映射模型，用作 CMM 能耗对照。"""
    base_model, _payload = load_baseline_template(args.baseline_checkpoint, device)
    crosssim_payload = torch.load(args.crosssim_only_checkpoint, map_location="cpu")
    if "crosssim_args" not in crosssim_payload:
        raise ValueError("CrossSim-only checkpoint missing crosssim_args.")
    if "crosssim_model_state_dict" not in crosssim_payload:
        raise ValueError("CrossSim-only checkpoint missing crosssim_model_state_dict.")
    crosssim_args = dict(crosssim_payload["crosssim_args"])
    crosssim_args["use_gpu"] = bool(args.use_gpu or device.type == "cuda")
    if crosssim_args["use_gpu"] and device.type != "cuda":
        raise RuntimeError("--use-gpu requires --device cuda.")
    model = build_model_from_crosssim_payload(base_model, crosssim_args, device)
    model.load_state_dict(crosssim_payload["crosssim_model_state_dict"])
    # 中文注释：load_state_dict 后必须把 PyTorch 权重同步回 CrossSim core，随后才能读取真实 matrix。
    synchronize_crosssim_cores(model)
    model.to(device)
    model.eval()
    return model, crosssim_payload


def is_mapped_linear(name: str, module: nn.Module) -> bool:
    """判断模块是否为计划范围内的 CrossSim AnalogLinear。"""
    if not (name.startswith("layers.") or name == "output_proj"):
        return False
    return hasattr(module, "core") and hasattr(module.core, "core") and hasattr(module.core.core, "cores")


def group_for_layer(layer_name: str) -> str:
    if layer_name == "output_proj":
        return "output_proj"
    if ".self_attn." in layer_name:
        return "self_attn"
    if ".cross_attn." in layer_name:
        return "cross_attn"
    if ".ffn." in layer_name:
        return "ffn"
    return "other"


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def g_raw_to_resistance(g_raw: np.ndarray, r_on: float, r_off: float) -> np.ndarray:
    """将 raw 归一化电导转为物理电阻，公式沿用 test_extract_conductance.py。"""
    g_physical = g_raw * (1.0 / r_on - 1.0 / r_off) + 1.0 / r_off
    if np.any(g_physical <= 0):
        raise ValueError("Non-positive physical conductance found after G_raw conversion.")
    return 1.0 / g_physical


def extract_tile_records(model: nn.Module, r_on: float, r_off: float) -> dict[str, TileRecord]:
    """从所有 mapped AnalogLinear 的 CrossSim core 中提取 tile 电阻矩阵。"""
    records: dict[str, TileRecord] = {}
    mapped_layers = [(name, module) for name, module in model.named_modules() if is_mapped_linear(name, module)]
    if len(mapped_layers) != 21:
        raise AssertionError(f"Expected 21 mapped linear layers, found {len(mapped_layers)}.")

    for layer_name, layer in mapped_layers:
        analog_core = layer.core.core
        for r, row_start, row_end in analog_core.row_partition_bounds:
            for c, col_start, col_end in analog_core.col_partition_bounds:
                wrapper = analog_core.cores[r][c]
                core_matrices: list[tuple[str, np.ndarray]] = []
                if hasattr(wrapper, "core") and hasattr(wrapper.core, "matrix"):
                    core_matrices.append(("single", to_numpy(wrapper.core.matrix)))
                elif hasattr(wrapper, "core_pos") and hasattr(wrapper, "core_neg"):
                    core_matrices.append(("pos", to_numpy(wrapper.core_pos.matrix)))
                    core_matrices.append(("neg", to_numpy(wrapper.core_neg.matrix)))
                else:
                    raise AttributeError(
                        f"Unsupported CrossSim core wrapper {type(wrapper).__name__}; "
                        "expected OffsetCore.core.matrix or BalancedCore.core_pos/core_neg.matrix."
                    )

                for polarity, g_raw in core_matrices:
                    r_physical = g_raw_to_resistance(g_raw, r_on, r_off)
                    # CrossSim matrix 方向是 weight[out, in]；SPICE 行电压对应 input feature，所以转置后统计。
                    conductance_by_input = 1.0 / r_physical.T
                    tile_id = f"{layer_name}|r{r}c{c}|{polarity}"
                    records[tile_id] = TileRecord(
                        tile_id=tile_id,
                        layer_name=layer_name,
                        group=group_for_layer(layer_name),
                        polarity=polarity,
                        core_row=int(r),
                        core_col=int(c),
                        row_start=int(row_start),
                        row_end=int(row_end),
                        col_start=int(col_start),
                        col_end=int(col_end),
                        g_raw=g_raw,
                        r_physical=r_physical,
                        conductance_by_input=conductance_by_input,
                    )
    return records


def infer_linear_features(layer: nn.Module) -> tuple[int, int]:
    """读取 AnalogLinear 的输入/输出维度，用于数字 MAC 对照。"""
    if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
        return int(layer.in_features), int(layer.out_features)
    if hasattr(layer, "core") and hasattr(layer.core, "core"):
        analog_core = layer.core.core
        return int(analog_core.ncol), int(analog_core.nrow)
    raise AttributeError(f"Cannot infer in/out features for {type(layer).__name__}.")


def extract_digital_mac_records(model: nn.Module) -> dict[str, DigitalMacRecord]:
    """为同一批 mapped Linear 建立数字 MAC 参考统计对象。"""
    records: dict[str, DigitalMacRecord] = {}
    mapped_layers = [(name, module) for name, module in model.named_modules() if is_mapped_linear(name, module)]
    if len(mapped_layers) != 21:
        raise AssertionError(f"Expected 21 mapped linear layers, found {len(mapped_layers)}.")
    for layer_name, layer in mapped_layers:
        in_features, out_features = infer_linear_features(layer)
        records[layer_name] = DigitalMacRecord(
            layer_name=layer_name,
            group=group_for_layer(layer_name),
            in_features=in_features,
            out_features=out_features,
        )
    return records


def scale_activation_vector(vector: np.ndarray, vread: float) -> np.ndarray:
    """activation_scale=per_vector：每个向量按自身最大绝对值缩放到 [-Vread, Vread]。"""
    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs == 0.0:
        return np.zeros_like(vector, dtype=np.float64)
    return vector.astype(np.float64) / max_abs * vread


def tile_power_from_voltage(record: TileRecord, row_voltages: np.ndarray) -> float:
    """P_tile = sum_i V_i^2 * sum_j G_ij，其中 i 是输入行。"""
    if row_voltages.size != record.conductance_by_input.shape[0]:
        raise ValueError(f"Voltage length mismatch for {record.tile_id}.")
    row_conductance_sum = record.conductance_by_input.sum(axis=1)
    return float(np.square(row_voltages).dot(row_conductance_sum))


def tile_read_current_from_voltage(record: TileRecord, row_voltages: np.ndarray) -> float:
    """统计输入侧总读电流；signed activation 下使用绝对行电流求和。"""
    row_conductance_sum = record.conductance_by_input.sum(axis=1)
    return float(np.abs(row_voltages * row_conductance_sum).sum())


def register_activation_hooks(
    model: nn.Module,
    records: dict[str, TileRecord],
    digital_mac_records: dict[str, DigitalMacRecord],
    pulse_ns: float,
    vread: float,
):
    """注册 hooks，在真实 forward 中把每个 mapped layer 的输入切到各个 tile 统计能耗。"""
    handles = []
    token_counter = {"count": 0}
    records_by_layer: dict[str, list[TileRecord]] = defaultdict(list)
    for record in records.values():
        records_by_layer[record.layer_name].append(record)

    def make_hook(layer_name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float().cpu()
            vectors = x.reshape(-1, x.shape[-1]).numpy()
            token_counter["count"] += int(vectors.shape[0])
            digital_mac_records[layer_name].vector_count += int(vectors.shape[0])
            for record in records_by_layer[layer_name]:
                tile_vectors = vectors[:, record.col_start : record.col_end]
                for vector in tile_vectors:
                    row_voltages = scale_activation_vector(vector, vread)
                    power_w = tile_power_from_voltage(record, row_voltages)
                    read_current_a = tile_read_current_from_voltage(record, row_voltages)
                    energy_j = power_w * pulse_ns * 1e-9
                    record.vector_count += 1
                    if power_w > 0.0:
                        record.nonzero_vector_count += 1
                    record.tile_total_power_w += power_w
                    record.tile_total_read_current_a += read_current_a
                    record.tile_total_energy_j += energy_j
                    record.vector_powers_w.append(power_w)
                    record.vector_energies_j.append(energy_j)
                    record.vector_voltages.append(row_voltages.astype(np.float64, copy=True))

        return hook

    for layer_name, module in model.named_modules():
        if layer_name in records_by_layer:
            handles.append(module.register_forward_hook(make_hook(layer_name)))
    return handles, token_counter


@torch.no_grad()
def run_activation_capture(
    model: nn.Module,
    loader,
    device: torch.device,
    limit_batches: int,
) -> int:
    """跑真实 COCO teacher-forcing forward，触发 hooks 采集 activation。"""
    model.eval()
    processed = 0
    for batch_idx, (images, captions) in enumerate(loader, start=1):
        if batch_idx > limit_batches:
            break
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        input_tokens = captions[:, :-1]
        _ = model(images, input_tokens)
        processed += int(images.shape[0])
    return processed


def estimate_read_energy_for_model(
    model: nn.Module,
    loader,
    device: torch.device,
    r_on: float,
    r_off: float,
    pulse_ns: float,
    vread: float,
    limit_batches: int,
) -> tuple[dict[str, TileRecord], dict[str, DigitalMacRecord], int, dict[str, int]]:
    """对一个 mapped CrossSim 模型统计 activation-aware read energy。"""
    records = extract_tile_records(model, r_on=r_on, r_off=r_off)
    digital_mac_records = extract_digital_mac_records(model)
    handles, token_counter = register_activation_hooks(
        model,
        records,
        digital_mac_records=digital_mac_records,
        pulse_ns=pulse_ns,
        vread=vread,
    )
    try:
        processed_images = run_activation_capture(
            model=model,
            loader=loader,
            device=device,
            limit_batches=limit_batches,
        )
    finally:
        for handle in handles:
            handle.remove()
    return records, digital_mac_records, processed_images, token_counter


def select_validation_tiles(records: dict[str, TileRecord]) -> dict[str, TileRecord]:
    """按统计窗口内 tile_total_energy 选择 high / median / low 非零能耗 tile。"""
    nonzero = [record for record in records.values() if record.tile_total_energy_j > 0.0]
    if not nonzero:
        raise RuntimeError("No nonzero-energy tile found; cannot select high/median/low.")
    ordered = sorted(nonzero, key=lambda record: record.tile_total_energy_j)
    return {
        "low": ordered[0],
        "median": ordered[len(ordered) // 2],
        "high": ordered[-1],
    }


def representative_vector(record: TileRecord) -> tuple[int, np.ndarray, float, float]:
    """选择最接近该 tile 平均单次功耗的一次 activation vector。"""
    if not record.vector_powers_w:
        raise RuntimeError(f"No vector powers recorded for {record.tile_id}.")
    mean_power = record.tile_mean_vector_power_w
    index = min(range(len(record.vector_powers_w)), key=lambda idx: abs(record.vector_powers_w[idx] - mean_power))
    return index, record.vector_voltages[index], record.vector_powers_w[index], record.vector_energies_j[index]


def digital_energy_j(total_macs: int, mac_energy_pj: float) -> float:
    """把 MAC 数转换为数字参考能耗。"""
    return float(total_macs) * mac_energy_pj * 1e-12


def pj_to_j(value_pj: float) -> float:
    """把 pJ 转换为 J。"""
    return float(value_pj) * 1e-12


def logical_tile_key(record: TileRecord) -> tuple[str, int, int]:
    """同一个 logical tile 可能对应 balanced 正负两个物理 core。"""
    return record.layer_name, record.core_row, record.core_col


def representative_records_by_logical_tile(records: dict[str, TileRecord]) -> dict[tuple[str, int, int], TileRecord]:
    """为每个 logical tile 选一个代表记录，外围 ADC/DAC 只在代表记录上计数。"""
    polarity_order = {"single": 0, "pos": 1, "neg": 2}
    representatives: dict[tuple[str, int, int], TileRecord] = {}
    for record in records.values():
        key = logical_tile_key(record)
        current = representatives.get(key)
        if current is None or polarity_order.get(record.polarity, 99) < polarity_order.get(current.polarity, 99):
            representatives[key] = record
    return representatives


def compute_energy_breakdowns(
    records: dict[str, TileRecord],
    model: PeripheralEnergyModel,
) -> dict[str, EnergyBreakdown]:
    """计算 peripheral-aware 分项。

    中文注释：core read energy 来自每个物理 core；ADC/DAC 和可选数字项按
    logical tile 计数，避免 balanced 正负阵列把外围转换次数重复计算。
    """
    breakdowns = {
        record.tile_id: EnergyBreakdown(core_read_energy_j=record.tile_total_energy_j)
        for record in records.values()
    }
    representatives = representative_records_by_logical_tile(records)

    layer_col_ids: dict[str, set[int]] = defaultdict(set)
    for record in representatives.values():
        layer_col_ids[record.layer_name].add(record.core_col)
    layer_min_col = {layer: min(cols) for layer, cols in layer_col_ids.items() if cols}
    layer_col_count = {layer: len(cols) for layer, cols in layer_col_ids.items()}

    for record in representatives.values():
        vector_count = float(record.vector_count)
        input_width = float(record.col_end - record.col_start)
        output_width = float(record.row_end - record.row_start)
        breakdown = breakdowns[record.tile_id]

        breakdown.adc_conversions = output_width * vector_count
        breakdown.dac_conversions = input_width * vector_count
        breakdown.adc_energy_j = breakdown.adc_conversions * pj_to_j(model.adc_energy_pj_per_conv)
        breakdown.dac_energy_j = breakdown.dac_conversions * pj_to_j(model.dac_energy_pj_per_conv)

        # partial-sum 累加和 bias 加法只按每个输出 row tile 计一次，挂到最小 col tile 的代表记录。
        if record.core_col == layer_min_col.get(record.layer_name):
            accum_per_output = max(layer_col_count.get(record.layer_name, 1) - 1, 0)
            breakdown.digital_accum_ops = output_width * vector_count * float(accum_per_output)
            breakdown.bias_ops = output_width * vector_count
            breakdown.digital_accum_energy_j = breakdown.digital_accum_ops * pj_to_j(model.digital_accum_energy_pj_per_op)
            breakdown.bias_energy_j = breakdown.bias_ops * pj_to_j(model.bias_energy_pj_per_op)

    return breakdowns


def sum_breakdowns(records: dict[str, TileRecord], breakdowns: dict[str, EnergyBreakdown], group: str) -> EnergyBreakdown:
    """按 group 汇总分项；group=full_decoder 时汇总全部 mapped Linear。"""
    total = EnergyBreakdown()
    for record in records.values():
        if group == "full_decoder" or record.group == group:
            total.add(breakdowns[record.tile_id])
    return total


def breakdown_to_columns(prefix: str, breakdown: EnergyBreakdown) -> dict[str, float]:
    """把能耗分项展开成 CSV 列。"""
    return {
        f"{prefix}core_read_energy_j": breakdown.core_read_energy_j,
        f"{prefix}adc_energy_j": breakdown.adc_energy_j,
        f"{prefix}dac_energy_j": breakdown.dac_energy_j,
        f"{prefix}digital_accum_energy_j": breakdown.digital_accum_energy_j,
        f"{prefix}bias_energy_j": breakdown.bias_energy_j,
        f"{prefix}adc_conversions": breakdown.adc_conversions,
        f"{prefix}dac_conversions": breakdown.dac_conversions,
        f"{prefix}digital_accum_ops": breakdown.digital_accum_ops,
        f"{prefix}bias_ops": breakdown.bias_ops,
    }


def write_summary_csv(
    path: Path,
    records: dict[str, TileRecord],
    breakdowns: dict[str, EnergyBreakdown],
    digital_mac_records: dict[str, DigitalMacRecord],
    metadata: dict[str, Any],
    crosssim_only_records: dict[str, TileRecord] | None = None,
    crosssim_only_breakdowns: dict[str, EnergyBreakdown] | None = None,
) -> None:
    group_power: dict[str, float] = defaultdict(float)
    group_read_current: dict[str, float] = defaultdict(float)
    for record in records.values():
        group_power[record.group] += record.tile_total_power_w
        group_read_current[record.group] += record.tile_total_read_current_a

    total_power = sum(record.tile_total_power_w for record in records.values())
    total_read_current = sum(record.tile_total_read_current_a for record in records.values())
    mac_energy_pj = float(metadata["digital_mac_energy_pj"])
    group_macs: dict[str, int] = defaultdict(int)
    for record in digital_mac_records.values():
        group_macs[record.group] += record.total_macs
    total_macs = sum(record.total_macs for record in digital_mac_records.values())

    crosssim_group_power: dict[str, float] = defaultdict(float)
    crosssim_group_read_current: dict[str, float] = defaultdict(float)
    if crosssim_only_records is not None:
        for record in crosssim_only_records.values():
            crosssim_group_power[record.group] += record.tile_total_power_w
            crosssim_group_read_current[record.group] += record.tile_total_read_current_a
    crosssim_total_power = (
        sum(record.tile_total_power_w for record in crosssim_only_records.values())
        if crosssim_only_records is not None
        else 0.0
    )
    crosssim_total_read_current = (
        sum(record.tile_total_read_current_a for record in crosssim_only_records.values())
        if crosssim_only_records is not None
        else 0.0
    )
    rows = []
    for group in ["self_attn", "cross_attn", "ffn", "output_proj", "full_decoder"]:
        cmm_breakdown = sum_breakdowns(records, breakdowns, group)
        cmm_energy = cmm_breakdown.total_energy_j
        crosssim_breakdown = (
            sum_breakdowns(crosssim_only_records, crosssim_only_breakdowns, group)
            if crosssim_only_records is not None and crosssim_only_breakdowns is not None
            else None
        )
        crosssim_energy = crosssim_breakdown.total_energy_j if crosssim_breakdown is not None else 0.0
        digital_macs = total_macs if group == "full_decoder" else group_macs.get(group, 0)
        digital_energy = digital_energy_j(digital_macs, mac_energy_pj)
        row = {
            **metadata,
            "group": group,
            "energy_j": cmm_energy,
            **breakdown_to_columns("", cmm_breakdown),
            "total_vector_power_w": total_power if group == "full_decoder" else group_power.get(group, 0.0),
            "total_vector_read_current_a": total_read_current
            if group == "full_decoder"
            else group_read_current.get(group, 0.0),
            "crosssim_only_energy_j": crosssim_energy if crosssim_breakdown is not None else "",
            **(
                breakdown_to_columns("crosssim_only_", crosssim_breakdown)
                if crosssim_breakdown is not None
                else {
                    "crosssim_only_core_read_energy_j": "",
                    "crosssim_only_adc_energy_j": "",
                    "crosssim_only_dac_energy_j": "",
                    "crosssim_only_digital_accum_energy_j": "",
                    "crosssim_only_bias_energy_j": "",
                    "crosssim_only_adc_conversions": "",
                    "crosssim_only_dac_conversions": "",
                    "crosssim_only_digital_accum_ops": "",
                    "crosssim_only_bias_ops": "",
                }
            ),
            "crosssim_only_total_vector_power_w": (
                crosssim_total_power
                if group == "full_decoder"
                else crosssim_group_power.get(group, 0.0)
            )
            if crosssim_only_records is not None
            else "",
            "crosssim_only_total_vector_read_current_a": (
                crosssim_total_read_current
                if group == "full_decoder"
                else crosssim_group_read_current.get(group, 0.0)
            )
            if crosssim_only_records is not None
            else "",
            "cmm_to_crosssim_only_energy_ratio": (
                cmm_energy / crosssim_energy if crosssim_breakdown is not None and crosssim_energy > 0 else ""
            ),
            "digital_mac_count": digital_macs,
            "digital_mac_energy_j": digital_energy,
            "cmm_to_digital_mac_energy_ratio": cmm_energy / digital_energy if digital_energy > 0 else "",
            "crosssim_only_to_digital_mac_energy_ratio": (
                crosssim_energy / digital_energy if crosssim_breakdown is not None and digital_energy > 0 else ""
            ),
        }
        rows.append(row)
    write_dict_rows(path, rows)


def write_digital_mac_reference_csv(
    path: Path,
    digital_mac_records: dict[str, DigitalMacRecord],
    metadata: dict[str, Any],
) -> None:
    """写出逐 mapped Linear 的数字 MAC 能耗参考。"""
    mac_energy_pj = float(metadata["digital_mac_energy_pj"])
    rows = []
    for record in sorted(digital_mac_records.values(), key=lambda item: item.layer_name):
        rows.append(
            {
                **metadata,
                "layer_name": record.layer_name,
                "group": record.group,
                "in_features": record.in_features,
                "out_features": record.out_features,
                "activation_vector_count": record.vector_count,
                "macs_per_vector": record.macs_per_vector,
                "digital_mac_count": record.total_macs,
                "digital_mac_energy_j": digital_energy_j(record.total_macs, mac_energy_pj),
            }
        )
    write_dict_rows(path, rows)


def write_tile_detail_csv(
    path: Path,
    records: dict[str, TileRecord],
    breakdowns: dict[str, EnergyBreakdown],
    ranks: dict[str, TileRecord],
    metadata: dict[str, Any],
) -> None:
    rank_by_id = {record.tile_id: rank for rank, record in ranks.items()}
    rows = []
    for record in sorted(records.values(), key=lambda item: item.tile_id):
        breakdown = breakdowns[record.tile_id]
        rows.append(
            {
                **metadata,
                "tile_rank": rank_by_id.get(record.tile_id, ""),
                "tile_id": record.tile_id,
                "layer_name": record.layer_name,
                "group": record.group,
                "polarity": record.polarity,
                "core_row": record.core_row,
                "core_col": record.core_col,
                "row_start": record.row_start,
                "row_end": record.row_end,
                "col_start": record.col_start,
                "col_end": record.col_end,
                "g_raw_min": float(record.g_raw.min()),
                "g_raw_max": float(record.g_raw.max()),
                "g_raw_mean": float(record.g_raw.mean()),
                "r_min_ohm": float(record.r_physical.min()),
                "r_max_ohm": float(record.r_physical.max()),
                "r_mean_ohm": float(record.r_physical.mean()),
                "tile_total_energy_j": breakdown.total_energy_j,
                **breakdown_to_columns("tile_", breakdown),
                "tile_total_read_current_a": record.tile_total_read_current_a,
                "tile_mean_vector_power_w": record.tile_mean_vector_power_w,
                "tile_mean_vector_read_current_a": record.tile_mean_vector_read_current_a,
                "vector_count": record.vector_count,
                "nonzero_vector_count": record.nonzero_vector_count,
            }
        )
    write_dict_rows(path, rows)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_spice_validation(
    ranks: dict[str, TileRecord],
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """为 high / median / low tile 的代表向量生成 ngspice 验证。"""
    rows: list[dict[str, Any]] = []
    validation_dir = args.output_dir / "spice_validation"
    for rank, record in ranks.items():
        vector_index, row_voltages, analytic_power_w, analytic_energy_j = representative_vector(record)
        # SPICE 行=input feature，列=output feature；CrossSim matrix 原方向为 output x input。
        resistance_for_spice = record.r_physical.T.tolist()
        tag = f"{rank}_{record.layer_name.replace('.', '_')}_r{record.core_row}c{record.core_col}"
        spice = run_matrix_experiment(
            resistance_ohm=resistance_for_spice,
            row_voltages=row_voltages.tolist(),
            pulse_ns=args.pulse_ns,
            ngspice=args.ngspice,
            out_dir=validation_dir,
            tag=tag,
            tran_step_ns=args.spice_tran_step_ns,
            pulse_rise_ns=args.spice_pulse_rise_ns,
            pulse_fall_ns=args.spice_pulse_fall_ns,
        )
        spice_energy = float(spice["energy_j"])
        rel_error = abs(spice_energy - analytic_energy_j) / max(abs(analytic_energy_j), 1e-30)
        validation_threshold = 1e-3 if (float(spice["pulse_rise_ns"]) > 0 or float(spice["pulse_fall_ns"]) > 0) else 1e-4
        rows.append(
            {
                **metadata,
                "tile_rank": rank,
                "tile_id": record.tile_id,
                "layer_name": record.layer_name,
                "polarity": record.polarity,
                "core_row": record.core_row,
                "core_col": record.core_col,
                "representative_vector_index": vector_index,
                "representative_vector_power_w": analytic_power_w,
                "representative_vector_analytic_energy_j": analytic_energy_j,
                "spice_integrated_energy_j": spice_energy,
                "relative_error": rel_error,
                "tran_step_ns": spice["tran_step_ns"],
                "pulse_rise_ns": spice["pulse_rise_ns"],
                "pulse_fall_ns": spice["pulse_fall_ns"],
                "validation_threshold_default": 1e-4,
                "validation_threshold_with_default_pulse_edges": 1e-3,
                "validation_pass": rel_error <= validation_threshold,
                "netlist": spice["netlist"],
                "waveform_csv": spice["waveform_csv"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, base_payload, crosssim_payload = load_cmm_crosssim_model(args, device)
    cmm_args = crosssim_payload["cmm_crosssim_args"]
    r_on = float(cmm_args.get("rmin", 1e3))
    r_off = float(cmm_args.get("rmax", 1e5))

    if args.coco_root is None:
        from src.coco_preprocess.loader import DEFAULT_COCO_ROOT

        coco_root = DEFAULT_COCO_ROOT
    else:
        coco_root = args.coco_root
    vocab = build_vocab_from_payload_or_data(base_payload, coco_root)
    max_len = int(base_payload["model_config"]["max_len"])
    loader = build_val_loader(
        coco_root=coco_root,
        vocab=vocab,
        max_len=max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.limit,
    )
    limit_batches = math.ceil(args.limit / max(args.batch_size, 1))

    records, digital_mac_records, processed_images, token_counter = estimate_read_energy_for_model(
        model=model,
        loader=loader,
        device=device,
        r_on=r_on,
        r_off=r_off,
        pulse_ns=args.pulse_ns,
        vread=args.vread,
        limit_batches=limit_batches,
    )
    peripheral_model = PeripheralEnergyModel(
        adc_energy_pj_per_conv=args.adc_energy_pj_per_conv,
        dac_energy_pj_per_conv=args.dac_energy_pj_per_conv,
        digital_accum_energy_pj_per_op=args.digital_accum_energy_pj_per_op,
        bias_energy_pj_per_op=args.bias_energy_pj_per_op,
    )
    breakdowns = compute_energy_breakdowns(records, peripheral_model)
    crosssim_only_records: dict[str, TileRecord] | None = None
    crosssim_only_breakdowns: dict[str, EnergyBreakdown] | None = None
    crosssim_only_ranks: dict[str, TileRecord] = {}
    crosssim_only_processed_images = 0
    crosssim_only_token_counter = {"count": 0}
    if not args.skip_crosssim_only:
        # 中文注释：释放 CMM 模型后再加载 CrossSim-only，降低大 checkpoint 同时驻留的内存压力。
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        crosssim_only_model, crosssim_only_payload = load_crosssim_only_model(args, device)
        crosssim_only_args = crosssim_only_payload["crosssim_args"]
        crosssim_only_records, _crosssim_only_mac_records, crosssim_only_processed_images, crosssim_only_token_counter = (
            estimate_read_energy_for_model(
                model=crosssim_only_model,
                loader=loader,
                device=device,
                r_on=float(crosssim_only_args.get("rmin", 1e3)),
                r_off=float(crosssim_only_args.get("rmax", 1e5)),
                pulse_ns=args.pulse_ns,
                vread=args.vread,
                limit_batches=limit_batches,
            )
        )
        crosssim_only_breakdowns = compute_energy_breakdowns(crosssim_only_records, peripheral_model)
        crosssim_only_ranks = select_validation_tiles(crosssim_only_records)
        del crosssim_only_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranks = select_validation_tiles(records)
    activation_vectors_per_tile = max((record.vector_count for record in records.values()), default=0)
    metadata = {
        "activation_scale": ACTIVATION_SCALE,
        "energy_scope": ENERGY_SCOPE,
        "excluded": EXCLUDED,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "cmm_crosssim_checkpoint": str(args.cmm_crosssim_checkpoint),
        "crosssim_only_checkpoint": str(args.crosssim_only_checkpoint) if not args.skip_crosssim_only else "",
        "vread_v": args.vread,
        "pulse_ns": args.pulse_ns,
        "sample_limit": args.limit,
        "processed_images": processed_images,
        "crosssim_only_processed_images": crosssim_only_processed_images if not args.skip_crosssim_only else "",
        "activation_vectors_per_tile": activation_vectors_per_tile,
        "token_vectors_seen_per_layer_sum": token_counter["count"],
        "crosssim_only_token_vectors_seen_per_layer_sum": (
            crosssim_only_token_counter["count"] if not args.skip_crosssim_only else ""
        ),
        "r_on_ohm": r_on,
        "r_off_ohm": r_off,
        "tile_shape": "x".join(str(v) for v in cmm_args.get("tile_shape", (128, 128))),
        "adc_resolution": cmm_args.get("adc_resolution", 0),
        "dac_resolution": cmm_args.get("dac_resolution", 0),
        "read_noise_std": cmm_args.get("read_noise_std", 0.0),
        "write_noise_std": cmm_args.get("write_noise_std", 0.0),
        "digital_reference_scope": "same_decoder_mapped_linear_mac_only",
        "digital_mac_energy_pj": args.digital_mac_energy_pj,
        "digital_mac_model": "configurable_reference",
        "adc_energy_pj_per_conv": args.adc_energy_pj_per_conv,
        "dac_energy_pj_per_conv": args.dac_energy_pj_per_conv,
        "digital_accum_energy_pj_per_op": args.digital_accum_energy_pj_per_op,
        "bias_energy_pj_per_op": args.bias_energy_pj_per_op,
        "peripheral_energy_source": args.peripheral_energy_source,
    }

    write_summary_csv(
        args.output_dir / "summary.csv",
        records,
        breakdowns,
        digital_mac_records,
        metadata,
        crosssim_only_records=crosssim_only_records,
        crosssim_only_breakdowns=crosssim_only_breakdowns,
    )
    write_tile_detail_csv(args.output_dir / "tile_detail.csv", records, breakdowns, ranks, metadata)
    if crosssim_only_records is not None:
        write_tile_detail_csv(
            args.output_dir / "crosssim_only_tile_detail.csv",
            crosssim_only_records,
            crosssim_only_breakdowns,
            crosssim_only_ranks,
            metadata,
        )
    write_digital_mac_reference_csv(args.output_dir / "digital_mac_reference.csv", digital_mac_records, metadata)

    spice_rows: list[dict[str, Any]] = []
    if not args.skip_ngspice:
        spice_rows = run_spice_validation(ranks, args, metadata)
        write_dict_rows(args.output_dir / "spice_validation.csv", spice_rows)
    else:
        (args.output_dir / "spice_validation.csv").write_text("", encoding="utf-8")

    manifest = {
        "metadata": metadata,
        "num_tiles": len(records),
        "selected_tiles": {rank: record.tile_id for rank, record in ranks.items()},
        "energy_model": {
            "scope": ENERGY_SCOPE,
            "formula": "core_read_energy_j + adc_energy_j + dac_energy_j + digital_accum_energy_j + bias_energy_j",
            "adc_energy_pj_per_conv": args.adc_energy_pj_per_conv,
            "dac_energy_pj_per_conv": args.dac_energy_pj_per_conv,
            "digital_accum_energy_pj_per_op": args.digital_accum_energy_pj_per_op,
            "bias_energy_pj_per_op": args.bias_energy_pj_per_op,
            "digital_mac_energy_pj": args.digital_mac_energy_pj,
            "source": args.peripheral_energy_source,
            "digital_reference_scope": "same_decoder_mapped_linear_mac_only",
        },
        "outputs": {
            "summary_csv": str(args.output_dir / "summary.csv"),
            "tile_detail_csv": str(args.output_dir / "tile_detail.csv"),
            "crosssim_only_tile_detail_csv": str(args.output_dir / "crosssim_only_tile_detail.csv")
            if crosssim_only_records is not None
            else "",
            "digital_mac_reference_csv": str(args.output_dir / "digital_mac_reference.csv"),
            "spice_validation_csv": str(args.output_dir / "spice_validation.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total_breakdown = sum_breakdowns(records, breakdowns, "full_decoder")
    total_energy = total_breakdown.total_energy_j
    total_crosssim_only_breakdown = (
        sum_breakdowns(crosssim_only_records, crosssim_only_breakdowns, "full_decoder")
        if crosssim_only_records is not None and crosssim_only_breakdowns is not None
        else None
    )
    total_crosssim_only_energy = total_crosssim_only_breakdown.total_energy_j if total_crosssim_only_breakdown is not None else 0.0
    total_digital_macs = sum(record.total_macs for record in digital_mac_records.values())
    total_digital_energy = digital_energy_j(total_digital_macs, args.digital_mac_energy_pj)
    print(f"mapped tiles: {len(records)}")
    print(f"processed images: {processed_images}")
    print(f"full decoder peripheral-aware energy_j: {total_energy:.6e}")
    print(f"  core_read_energy_j: {total_breakdown.core_read_energy_j:.6e}")
    print(f"  adc_energy_j: {total_breakdown.adc_energy_j:.6e}")
    print(f"  dac_energy_j: {total_breakdown.dac_energy_j:.6e}")
    print(f"  digital_accum_energy_j: {total_breakdown.digital_accum_energy_j:.6e}")
    print(f"  bias_energy_j: {total_breakdown.bias_energy_j:.6e}")
    if total_crosssim_only_breakdown is not None:
        ratio = total_energy / total_crosssim_only_energy if total_crosssim_only_energy > 0 else float("nan")
        print(f"crosssim-only full decoder peripheral-aware energy_j: {total_crosssim_only_energy:.6e}")
        print(f"cmm/crosssim-only energy ratio: {ratio:.6e}")
    print(f"digital mac count: {total_digital_macs}")
    print(f"digital mac reference energy_j: {total_digital_energy:.6e}")
    print(f"cmm/digital mac energy ratio: {total_energy / total_digital_energy:.6e}")
    print(f"summary: {args.output_dir / 'summary.csv'}")
    print(f"tile detail: {args.output_dir / 'tile_detail.csv'}")
    print(f"digital MAC reference: {args.output_dir / 'digital_mac_reference.csv'}")
    if spice_rows:
        max_rel_error = max(row["relative_error"] for row in spice_rows)
        print(f"spice validation: {args.output_dir / 'spice_validation.csv'}")
        print(f"max spice relative error: {max_rel_error:.6e}")


if __name__ == "__main__":
    main()
