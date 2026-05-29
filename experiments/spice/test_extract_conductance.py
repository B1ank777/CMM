"""测试从 CrossSim AnalogLinear 的 core 中提取实际 conductance / resistance array。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from simulator.algorithms.dnn.torch import AnalogLinear, from_torch, synchronize
from simulator.algorithms.dnn.analog_linear import AnalogLinear as LinearCore  # dnn wrapper
from simulator.cores.analog_core import AnalogCore
from simulator import CrossSimParameters


def build_crosssim_params(
    tile_shape=(64, 64),
    adc_resolution=0,
    dac_resolution=0,
    rmin=1e3,
    rmax=1e5,
    cell_bits=0,
    read_noise_std=0.0,
    programming_error_std=0.0,
):
    """构建 CrossSimParameters (与 map_crosssim.build_crosssim_params 等同)。"""
    params = CrossSimParameters()
    params.update(
        {
            "core.style": "OFFSET",
            "core.offset.style": "DIGITAL_OFFSET",
            "core.rows_max": int(tile_shape[0]),
            "core.cols_max": int(tile_shape[1]),
            "core.mapping.weights.percentile": None,
            "core.mapping.weights.min": -1.0,
            "core.mapping.weights.max": 1.0,
            "core.output_dtype": "FLOAT32",
            "simulation.useGPU": False,
            "simulation.disable_fast_matmul": True,
            "simulation.convolution.x_par": 1,
            "simulation.convolution.y_par": 1,
            "xbar.device.Rmin": float(rmin),
            "xbar.device.Rmax": float(rmax),
            "xbar.device.cell_bits": int(cell_bits),
            "xbar.adc.mvm.model": "QuantizerADC" if adc_resolution > 0 else "IdealADC",
            "xbar.adc.vmm.model": "QuantizerADC" if adc_resolution > 0 else "IdealADC",
            "xbar.adc.mvm.bits": int(adc_resolution),
            "xbar.adc.vmm.bits": int(adc_resolution),
            "xbar.adc.mvm.adc_range_option": "MAX",
            "xbar.adc.vmm.adc_range_option": "MAX",
            "xbar.dac.mvm.model": "QuantizerDAC" if dac_resolution > 0 else "IdealDAC",
            "xbar.dac.vmm.model": "QuantizerDAC" if dac_resolution > 0 else "IdealDAC",
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


def extract_raw_conductance(layer: AnalogLinear) -> dict:
    """从 CrossSim AnalogLinear 中提取最底层的 raw conductance matrix。

    访问路径: layer.core (nn -> dnn).core (dnn -> AnalogCore).cores[r][c] (OffsetCore).core (NumericCore).matrix

    返回 dict:
        - "G_raw": 原始电导矩阵 (已受 write_noise / programming error 影响)
        - "nrow", "ncol": 总行列
        - "num_cores_row", "num_cores_col": 分区数
        - "cores_info": 每个子核的详细信息和 G 矩阵
    """
    dnn_core: LinearCore = layer.core
    analog_core: AnalogCore = dnn_core.core

    results = {
        "G_raw": None,
        "num_cores_row": analog_core.num_cores_row,
        "num_cores_col": analog_core.num_cores_col,
        "nrow": analog_core.nrow,
        "ncol": analog_core.ncol,
        "shape": analog_core.shape,
        "cores_info": [],
    }

    full_matrix = np.zeros((analog_core.nrow, analog_core.ncol))
    for r, row_start, row_end in analog_core.row_partition_bounds:
        for c, col_start, col_end in analog_core.col_partition_bounds:
            wrapper = analog_core.cores[r][c]   # OffsetCore
            numeric = wrapper.core               # NumericCore
            G = numeric.matrix                   # raw conductance matrix
            if hasattr(G, "get"):                # CuPy -> NumPy
                G = G.get()
            G = np.asarray(G)

            info = {
                "r": r, "c": c,
                "row_start": int(row_start), "col_start": int(col_start),
                "G": G, "G_shape": G.shape,
                "G_min": float(G.min()), "G_max": float(G.max()),
                "G_mean": float(G.mean()), "G_std": float(G.std()),
            }
            results["cores_info"].append(info)
            full_matrix[row_start:row_end, col_start:col_end] = G

    results["G_raw"] = full_matrix
    return results


def G_to_weight(G: np.ndarray, Gmin_norm: float, Wrange_xbar: float, weight_max: float) -> np.ndarray:
    """OffsetCore._wrapper_read_matrix 的逆操作:
    weight = ((G - Gmin_norm) / Wrange_xbar - 0.5) * 2 * max
    (对 DIGITAL_OFFSET 模式)
    """
    return ((G - Gmin_norm) / Wrange_xbar - 0.5) * 2 * weight_max


def main():
    print("=" * 65)
    print("Test: Extract raw conductance from CrossSim AnalogLinear cores")
    print("=" * 65)

    torch.manual_seed(42)

    # 1. Build a simple nn.Linear and convert it to AnalogLinear
    print("\n[1] Building nn.Linear(64, 128) -> AnalogLinear")
    digital = nn.Linear(64, 128).eval()

    params = build_crosssim_params(
        tile_shape=(64, 64),
        adc_resolution=0,       # ideal ADC (no quantization)
        dac_resolution=0,       # ideal DAC
        cell_bits=0,            # continuous conductance
    )

    analog = from_torch(digital, params, bias_rows=0).eval()
    synchronize(analog)

    print(f"    in_features={analog.in_features}, out_features={analog.out_features}")
    print(f"    Type of analog.core: {type(analog.core).__name__}")
    dnn_core = analog.core
    print(f"    Type of analog.core.core: {type(dnn_core.core).__name__}")
    analog_core = dnn_core.core
    print(f"    Ncores: {analog_core.Ncores}")
    print(f"    cores grid: {analog_core.num_cores_row}x{analog_core.num_cores_col}")
    wrapper = analog_core.cores[0][0]
    print(f"    Type of cores[0][0]: {type(wrapper).__name__}")
    print(f"    Type of cores[0][0].core (NumericCore): {type(wrapper.core).__name__}")

    # 2. Extract raw conductance
    print("\n[2] Extracting raw conductance...")
    results = extract_raw_conductance(analog)
    print(f"    Full G_raw shape: {results['G_raw'].shape}")
    print(f"    G_raw range: [{results['G_raw'].min():.6f}, {results['G_raw'].max():.6f}]")
    for info in results["cores_info"]:
        print(f"    core[{info['r']}][{info['c']}]: shape={info['G_shape']}, "
              f"G∈[{info['G_min']:.6f}, {info['G_max']:.6f}]")

    # 3. Verify round-trip: G -> weight -> match with nn.Linear.weight
    print("\n[3] Round-trip verification: raw G -> reconstructed weight")

    # Get the actual weight via AnalogCore.get_matrix()
    recovered_weight_via_api = analog_core.get_matrix()
    recovered_weight_via_api = np.asarray(recovered_weight_via_api)

    # Get Gmin_norm / Wrange_xbar from the actual wrapper core
    dnn_core = analog.core
    analog_core_inner = dnn_core.core
    wrapper = analog_core_inner.cores[0][0]
    Gmin_norm = wrapper.Gmin_norm
    Wrange_xbar = wrapper.Wrange_xbar
    weight_max = analog_core_inner.params.core.mapping.weights.max
    print(f"    Gmin_norm={Gmin_norm}, Wrange_xbar={Wrange_xbar}, weight_max={weight_max}")

    # Manual reconstruction
    G_raw = results["G_raw"]
    manual_recovered = G_to_weight(G_raw, Gmin_norm, Wrange_xbar, weight_max)

    digital_weight = digital.weight.detach().cpu().numpy()

    diff_api = np.abs(recovered_weight_via_api - digital_weight)
    diff_manual = np.abs(manual_recovered - digital_weight)

    print(f"    Via get_matrix():  MAE={diff_api.mean():.10f}, MaxErr={diff_api.max():.10f}")
    print(f"    Manual from raw G: MAE={diff_manual.mean():.10f}, MaxErr={diff_manual.max():.10f}")

    # 4. Test with cell_bits quantization (non-trivial case)
    print("\n[4] Test with cell_bits=8 (conductance quantization)")
    params8 = build_crosssim_params(
        tile_shape=(64, 64),
        cell_bits=8,
    )
    analog8 = from_torch(digital, params8, bias_rows=0).eval()
    synchronize(analog8)
    results8 = extract_raw_conductance(analog8)

    print(f"    Full G_raw shape: {results8['G_raw'].shape}")
    print(f"    G_raw range: [{results8['G_raw'].min():.6f}, {results8['G_raw'].max():.6f}]")
    print(f"    Unique G values: {len(np.unique(np.round(results8['G_raw'], 8)))}")
    for info in results8["cores_info"]:
        n_unique = len(np.unique(np.round(info["G"], 8)))
        print(f"    core[{info['r']}][{info['c']}]: G∈[{info['G_min']:.6f}, {info['G_max']:.6f}], "
              f"{n_unique} unique values")

    recovered8 = analog8.core.core.get_matrix()
    recovered8 = np.asarray(recovered8)
    diff8 = np.abs(recovered8 - digital_weight)
    print(f"    cell_bits=8: recovered vs digital MAE={diff8.mean():.6f}, MaxErr={diff8.max():.6f}")

    # 5. Test writing noise (programming error)
    print("\n[5] Test with programming_error_std=0.01 (write noise)")
    params_w = build_crosssim_params(
        tile_shape=(64, 64),
        programming_error_std=0.01,
    )
    analog_w = from_torch(digital, params_w, bias_rows=0).eval()
    synchronize(analog_w)
    results_w = extract_raw_conductance(analog_w)

    for info in results_w["cores_info"]:
        print(f"    core[{info['r']}][{info['c']}]: G∈[{info['G_min']:.6f}, {info['G_max']:.6f}], "
              f"mean={info['G_mean']:.6f}, std={info['G_std']:.6f}")

    recovered_w = analog_w.core.core.get_matrix()
    recovered_w = np.asarray(recovered_w)
    diff_w = np.abs(recovered_w - digital_weight)
    print(f"    write_noise=0.01: recovered vs digital MAE={diff_w.mean():.6f}, MaxErr={diff_w.max():.6f}")

    # 6. Physical resistance estimation
    print("\n[6] Physical resistance estimation")
    # CrossSim 默认 VTEAM model: w_max = -10.0 → high-R state, w_min = 0.0 → low-R state
    # 归一化电导 G_norm 范围 [Gmin_norm, Gmax_norm] = [0, 1], 其中:
    #   G_norm = 0 → 最高电阻 (= R_off ≈ 100kΩ)
    #   G_norm = 1 → 最低电阻 (= R_on ≈ 1kΩ)
    # 反推: R = 1 / (G_norm * (1/R_on - 1/R_off) + 1/R_off)

    R_on = analog.core.core.params.xbar.device.Rmin
    R_off = analog.core.core.params.xbar.device.Rmax
    G_on = 1.0 / R_on
    G_off = 1.0 / R_off
    print(f"    R_on={R_on}Ω, R_off={R_off}Ω")
    print(f"    G_on={G_on}, G_off={G_off}")

    G_raw_sample = results["G_raw"][:5, :5]
    print(f"    前 5x5 raw G:\n{G_raw_sample}")

    # 实际电导 = G_norm * (G_on - G_off) + G_off
    G_physical = G_raw_sample * (G_on - G_off) + G_off
    R_physical = 1.0 / G_physical
    print(f"    前 5x5 物理电阻 (Ω):\n{R_physical}")

    print("\n" + "=" * 65)
    print("RESULT: Successfully extracted raw conductance from every")
    print("CrossSim AnalogLinear core!")
    print()
    print("Access path:")
    print("  layer.core (AnalogLinear[torch]) ->")
    print("  layer.core.core (AnalogLinear[dnn] -> AnalogCore) ->")
    print("  layer.core.core.cores[r][c] (OffsetCore) ->")
    print("  layer.core.core.cores[r][c].core (NumericCore) ->")
    print("  layer.core.core.cores[r][c].core.matrix (ndarray)")
    print("=" * 65)


if __name__ == "__main__":
    main()
