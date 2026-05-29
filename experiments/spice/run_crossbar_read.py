"""基础 crossbar 读功耗 SPICE 实验工具。

运行示例:
    python experiments/spice/run_crossbar_read.py --n 16 --vread 0.1 --pulse-ns 10
    python experiments/spice/run_crossbar_read.py --n 32 --pattern sparse --high-ratio 0.3 --ron 1e3 --roff 1e5

说明:
    本文件既可以独立跑实验 A/B 的理想电阻阵列，也给
    estimate_cmm_crosssim_read_energy.py 提供 run_matrix_experiment()，
    用 CMM-CrossSim 提取出的真实 tile 电阻矩阵生成 ngspice 验证网表。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PULSE_RISE_NS = 0.1
DEFAULT_PULSE_FALL_NS = 0.1


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _default_tran_step_ns(pulse_ns: float) -> float:
    # 中文注释：默认给 200 个采样点，且不粗于 10 ps，减小 PULSE 边沿积分误差。
    return max(float(pulse_ns) / 200.0, 0.01)


def _safe_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _write_waveform_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "power_w"])
        writer.writeheader()
        writer.writerows(rows)


def analytic_matrix_energy(
    resistance_ohm: list[list[float]] | np.ndarray,
    row_voltages: list[float] | np.ndarray,
    pulse_ns: float,
) -> dict[str, float]:
    """按理想虚地列端解析计算 crossbar core read power/energy。"""
    resistance = np.asarray(resistance_ohm, dtype=np.float64)
    voltages = np.asarray(row_voltages, dtype=np.float64)
    conductance = 1.0 / resistance
    row_power = (voltages**2) * conductance.sum(axis=1)
    total_power = float(row_power.sum())
    total_current = float(np.sum(np.abs(voltages) * conductance.sum(axis=1)))
    return {
        "power_w": total_power,
        "energy_j": total_power * float(pulse_ns) * 1e-9,
        "read_current_a": total_current,
    }


def write_matrix_netlist(
    resistance_ohm: list[list[float]] | np.ndarray,
    row_voltages: list[float] | np.ndarray,
    pulse_ns: float,
    netlist_path: Path,
    dat_path: Path,
    tran_step_ns: float | None = None,
    pulse_rise_ns: float = DEFAULT_PULSE_RISE_NS,
    pulse_fall_ns: float = DEFAULT_PULSE_FALL_NS,
) -> dict[str, float]:
    """生成理想 crossbar 网表；行端 PULSE，列端 0 V 虚地。"""
    resistance = np.asarray(resistance_ohm, dtype=np.float64)
    voltages = np.asarray(row_voltages, dtype=np.float64)
    if resistance.ndim != 2:
        raise ValueError("resistance_ohm must be a 2-D matrix")
    if resistance.shape[0] != voltages.shape[0]:
        raise ValueError("row_voltages length must match resistance matrix row count")

    rows, cols = resistance.shape
    tran_step_ns = _default_tran_step_ns(pulse_ns) if tran_step_ns is None else float(tran_step_ns)
    stop_ns = float(pulse_ns) + float(pulse_rise_ns) + float(pulse_fall_ns)
    period_ns = max(stop_ns * 2.0, float(pulse_ns) * 3.0)

    netlist_path.parent.mkdir(parents=True, exist_ok=True)
    dat_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "* Ideal crossbar read-energy validation",
        f"* rows={rows} cols={cols} pulse_ns={_format_float(pulse_ns)}",
        "* Each row is collapsed to an equivalent conductance because columns are ideal virtual ground.",
    ]
    for i, voltage in enumerate(voltages):
        lines.append(
            "Vrow{idx} row{idx} 0 PULSE(0 {v} 0 {tr}n {tf}n {pw}n {per}n)".format(
                idx=i,
                v=_format_float(voltage),
                tr=_format_float(pulse_rise_ns),
                tf=_format_float(pulse_fall_ns),
                pw=_format_float(pulse_ns),
                per=_format_float(period_ns),
            )
        )
    power_terms: list[str] = []
    for i in range(rows):
        # 中文注释：列端为理想虚地时，同一行所有 cell 可等效为一个到地电导，能量积分完全一致。
        row_g = float(np.sum(1.0 / resistance[i, :]))
        req = 1.0 / max(row_g, 1e-30)
        lines.append(f"Reqrow{i} row{i} 0 {_format_float(req)}")
        power_terms.append(f"(v(row{i})*v(row{i})/{_format_float(req)})")

    expr = " + ".join(power_terms) if power_terms else "0"
    lines.extend(
        [
            f"Bpwr pwr 0 v = {expr}",
            ".control",
            "set filetype=ascii",
            f"tran {_format_float(tran_step_ns)}n {_format_float(stop_ns)}n",
            f"meas tran energy INTEG v(pwr) FROM=0n TO={_format_float(stop_ns)}n",
            f"wrdata {dat_path.as_posix()} v(pwr)",
            "quit",
            ".endc",
            ".end",
        ]
    )
    netlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "tran_step_ns": tran_step_ns,
        "pulse_rise_ns": float(pulse_rise_ns),
        "pulse_fall_ns": float(pulse_fall_ns),
        "stop_ns": stop_ns,
    }


def _parse_ngspice_energy(log_text: str) -> float | None:
    match = re.search(r"energy\s*=\s*([-+0-9.eE]+)", log_text)
    if not match:
        return None
    return float(match.group(1))


def _load_dat_power(dat_path: Path) -> list[dict[str, float]]:
    if not dat_path.exists():
        return []
    rows: list[dict[str, float]] = []
    for raw in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) >= 2:
            rows.append({"time_s": float(parts[0]), "power_w": float(parts[1])})
    return rows


def _integrate_power_rows(rows: list[dict[str, float]]) -> float | None:
    if len(rows) < 2:
        return None
    time = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    power = np.asarray([row["power_w"] for row in rows], dtype=np.float64)
    return float(np.trapezoid(power, time))


def run_matrix_experiment(
    resistance_ohm: list[list[float]] | np.ndarray,
    row_voltages: list[float] | np.ndarray,
    pulse_ns: float,
    ngspice: str = "ngspice",
    out_dir: str | Path = "experiments/spice/results/matrix",
    tag: str = "matrix",
    tran_step_ns: float | None = None,
    pulse_rise_ns: float = DEFAULT_PULSE_RISE_NS,
    pulse_fall_ns: float = DEFAULT_PULSE_FALL_NS,
) -> dict[str, Any]:
    """生成 .cir，调用 ngspice，并返回积分能量等验证信息。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_tag(tag)
    netlist = out_dir / f"{tag}.cir"
    dat = out_dir / f"{tag}.dat"
    waveform_csv = out_dir / f"{tag}.csv"
    log_path = out_dir / f"{tag}.log"

    timing = write_matrix_netlist(
        resistance_ohm=resistance_ohm,
        row_voltages=row_voltages,
        pulse_ns=pulse_ns,
        netlist_path=netlist,
        dat_path=dat,
        tran_step_ns=tran_step_ns,
        pulse_rise_ns=pulse_rise_ns,
        pulse_fall_ns=pulse_fall_ns,
    )
    analytic = analytic_matrix_energy(resistance_ohm, row_voltages, pulse_ns)

    cmd = [ngspice, "-b", str(netlist.resolve())]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    log_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    log_path.write_text(log_text, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        raise RuntimeError(f"ngspice failed for {netlist}; see {log_path}")

    rows = _load_dat_power(dat)
    _write_waveform_csv(waveform_csv, rows)
    energy = _parse_ngspice_energy(log_text)
    if energy is None:
        # 中文注释：Windows 版 ngspice 有时不打印 meas 结果，此时直接对 wrdata 波形做梯形积分。
        energy = _integrate_power_rows(rows)
    if energy is None:
        raise RuntimeError(f"ngspice did not produce an energy measure or waveform data; see {log_path}")

    return {
        "energy_j": float(energy),
        "analytic_energy_j": analytic["energy_j"],
        "avg_power_w": float(energy) / (float(pulse_ns) * 1e-9),
        "read_current_a": analytic["read_current_a"],
        "netlist": str(netlist),
        "dat": str(dat),
        "waveform_csv": str(waveform_csv),
        "log": str(log_path),
        **timing,
    }


def build_resistance_matrix(n: int, ron: float, roff: float, pattern: str, high_ratio: float) -> np.ndarray:
    """构造实验 A/B 的简化 memristor 电阻矩阵。"""
    if pattern == "all_on":
        return np.full((n, n), ron, dtype=np.float64)
    if pattern == "all_off":
        return np.full((n, n), roff, dtype=np.float64)
    if pattern == "checker":
        idx = np.indices((n, n)).sum(axis=0) % 2 == 0
        return np.where(idx, ron, roff).astype(np.float64)
    rng = np.random.default_rng(0)
    high = rng.random((n, n)) < float(high_ratio)
    return np.where(high, ron, roff).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ideal crossbar read-power ngspice experiment.")
    parser.add_argument("--n", type=int, default=16, choices=[16, 32], help="crossbar size")
    parser.add_argument("--vread", type=float, default=0.1, help="read voltage/V")
    parser.add_argument("--pulse-ns", type=float, default=10.0, help="read pulse width/ns")
    parser.add_argument("--ron", type=float, default=1e3, help="low-resistance state/ohm")
    parser.add_argument("--roff", type=float, default=1e5, help="high-resistance state/ohm")
    parser.add_argument("--pattern", choices=["all_on", "all_off", "checker", "sparse"], default="checker")
    parser.add_argument("--high-ratio", type=float, default=0.3, help="sparse 模式下 Ron 占比")
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/spice/results/basic_crossbar"))
    args = parser.parse_args()

    resistance = build_resistance_matrix(args.n, args.ron, args.roff, args.pattern, args.high_ratio)
    row_voltages = np.full(args.n, args.vread, dtype=np.float64)
    result = run_matrix_experiment(
        resistance_ohm=resistance,
        row_voltages=row_voltages,
        pulse_ns=args.pulse_ns,
        ngspice=args.ngspice,
        out_dir=args.out_dir,
        tag=f"{args.n}x{args.n}_{args.pattern}",
    )
    print(f"energy_j: {result['energy_j']:.6e}")
    print(f"avg_power_w: {result['avg_power_w']:.6e}")
    print(f"read_current_a: {result['read_current_a']:.6e}")
    print(f"netlist: {result['netlist']}")


if __name__ == "__main__":
    main()
