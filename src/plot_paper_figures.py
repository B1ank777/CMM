"""Generate paper figures for CMM non-ideality sensitivity experiments.

运行示例:
    python -m src.plot_paper_figures
    python -m src.plot_paper_figures --output-dir figures --formats png,pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE = {"bleu4": 0.2464, "cider": 0.8546}

# 真实实验结果，来源于 log/experiment_2026-05-25_cmm_dac_ablation.md。
DAC_SWEEP = {
    "bits": [12, 10, 8, 6, 4],
    "bleu4": [0.2478, 0.2496, 0.2462, 0.2472, 0.2309],
    "cider": [0.8557, 0.8605, 0.8523, 0.8536, 0.8107],
}

# 真实实验结果，来源于 log/experiment_2026-05-25_cmm_adc_ablation.md。
ADC_SWEEP = {
    "bits": [12, 10, 8, 6, 4],
    "bleu4": [0.2497, 0.2521, 0.2491, 0.2463, 0.2201],
    "cider": [0.8603, 0.8597, 0.8517, 0.8357, 0.7830],
}

# 真实实验结果，来源于 log/experiment_2026-05-25_cmm_cell_bits_ablation.md。
CELL_BITS_SWEEP = {
    "bits": [0, 2, 3, 4, 6, 8],
    "labels": ["cont.", "2", "3", "4", "6", "8"],
    "bleu4": [0.2464, 0.0, 0.1708, 0.2266, 0.2443, 0.2455],
    "cider": [0.8546, 0.0, 0.6624, 0.8018, 0.8377, 0.8503],
}

# 真实实验结果，来源于 log/experiment_2026-05-25_cmm_array_size_ablation.md。
# 注意：该结果非单调，因此保持真实折线波动，不做趋势化或平滑处理。
ARRAY_SIZE_SWEEP = {
    "sizes": [64, 128, 256, 512],
    "bleu4": [0.2479, 0.2491, 0.2470, 0.2484],
    "cider": [0.8524, 0.8517, 0.8543, 0.8544],
}

# 真实实验结果，来源于 log/experiment_2026-05-25_cmm_write_noise_ablation.md。
WRITE_NOISE_SWEEP = {
    "std": [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "bleu4": [0.2464, 0.2477, 0.2487, 0.2481, 0.2444, 0.2410],
    "bleu4_err": [0.0, 0.0017, 0.0012, 0.0019, 0.0035, 0.0054],
    "cider": [0.8546, 0.8548, 0.8539, 0.8540, 0.8490, 0.8395],
    "cider_err": [0.0, 0.0035, 0.0021, 0.0004, 0.0008, 0.0070],
}

# 真实实验结果，来源于 log/experiment_2026-05-25_cmm_read_noise_ablation.md。
READ_NOISE_SWEEP = {
    "std": [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "bleu4": [0.2464, 0.2464, 0.2472, 0.2490, 0.2492, 0.2479],
    "bleu4_err": [0.0, 0.0, 0.0007, 0.0018, 0.0022, 0.0022],
    "cider": [0.8546, 0.8547, 0.8556, 0.8590, 0.8588, 0.8505],
    "cider_err": [0.0, 0.0001, 0.0005, 0.0027, 0.0049, 0.0040],
}

# Fig. 7 只聚焦来源规划指定的三条最大 gap 路径。
GAP_DATA = [
    {
        "condition": "DAC-12",
        "cmm_bleu4": 0.2478,
        "crosssim_bleu4": 0.1614,
        "cmm_cider": 0.8557,
        "crosssim_cider": 0.5380,
    },
    {
        "condition": "ADC-8",
        "cmm_bleu4": 0.2491,
        "crosssim_bleu4": 0.2166,
        "cmm_cider": 0.8517,
        "crosssim_cider": 0.7542,
    },
    {
        "condition": "cell_bits-4",
        "cmm_bleu4": 0.2266,
        "crosssim_bleu4": 0.2262,
        "cmm_cider": 0.8018,
        "crosssim_cider": 0.8010,
    },
]


def configure_style() -> None:
    """设置论文图的统一风格。"""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    """保存 PNG/PDF 等多种格式。"""
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def plot_dual_metric_line(
    ax: plt.Axes,
    x: list[float],
    bleu4: list[float],
    cider: list[float],
    xlabel: str,
    title: str,
    xticklabels: list[str] | None = None,
    log_x: bool = False,
    bleu4_err: list[float] | None = None,
    cider_err: list[float] | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """在同一子图中用双纵轴画 BLEU-4 与 CIDEr。"""
    color_bleu = "#1f77b4"
    color_cider = "#d62728"

    ax2 = ax.twinx()
    if log_x:
        # log 轴不能显示 0，用一个很小的左端点表示 no noise，并单独标注。
        plot_x = [1e-5 if value == 0 else value for value in x]
        ax.set_xscale("log")
        ax.set_xticks(plot_x)
        ax.set_xticklabels(["0", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"])
    else:
        plot_x = x
        ax.set_xticks(plot_x)
        if xticklabels:
            ax.set_xticklabels(xticklabels)

    line_kwargs = {"linewidth": 1.8, "markersize": 4.5, "capsize": 3}
    ax.errorbar(
        plot_x,
        bleu4,
        yerr=bleu4_err,
        color=color_bleu,
        marker="o",
        label="BLEU-4",
        **line_kwargs,
    )
    ax2.errorbar(
        plot_x,
        cider,
        yerr=cider_err,
        color=color_cider,
        marker="s",
        label="CIDEr",
        **line_kwargs,
    )

    ax.axhline(BASELINE["bleu4"], color=color_bleu, linestyle="--", linewidth=0.9, alpha=0.45)
    ax2.axhline(BASELINE["cider"], color=color_cider, linestyle="--", linewidth=0.9, alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BLEU-4", color=color_bleu)
    ax2.set_ylabel("CIDEr", color=color_cider)
    ax.tick_params(axis="y", colors=color_bleu)
    ax2.tick_params(axis="y", colors=color_cider)
    ax.set_title(title)
    ax.grid(True, axis="both", alpha=0.25, linewidth=0.6)
    return ax, ax2


def make_fig4(output_dir: Path, formats: list[str]) -> None:
    """Fig. 4: DAC/ADC precision sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    plot_dual_metric_line(
        axes[0],
        DAC_SWEEP["bits"],
        DAC_SWEEP["bleu4"],
        DAC_SWEEP["cider"],
        "DAC bits",
        "DAC sweep",
    )
    plot_dual_metric_line(
        axes[1],
        ADC_SWEEP["bits"],
        ADC_SWEEP["bleu4"],
        ADC_SWEEP["cider"],
        "ADC bits",
        "ADC sweep",
    )
    save_figure(fig, output_dir, "fig4_dac_adc_precision_sensitivity", formats)


def make_fig5(output_dir: Path, formats: list[str]) -> None:
    """Fig. 5: cell_bits / array_size sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    plot_dual_metric_line(
        axes[0],
        CELL_BITS_SWEEP["bits"],
        CELL_BITS_SWEEP["bleu4"],
        CELL_BITS_SWEEP["cider"],
        "cell_bits",
        "cell_bits sweep",
        xticklabels=CELL_BITS_SWEEP["labels"],
    )
    plot_dual_metric_line(
        axes[1],
        ARRAY_SIZE_SWEEP["sizes"],
        ARRAY_SIZE_SWEEP["bleu4"],
        ARRAY_SIZE_SWEEP["cider"],
        "array_size",
        "array_size sweep",
        xticklabels=[str(size) for size in ARRAY_SIZE_SWEEP["sizes"]],
    )
    save_figure(fig, output_dir, "fig5_cell_bits_array_size_sensitivity", formats)


def make_fig6(output_dir: Path, formats: list[str]) -> None:
    """Fig. 6: write_noise / read_noise sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    plot_dual_metric_line(
        axes[0],
        WRITE_NOISE_SWEEP["std"],
        WRITE_NOISE_SWEEP["bleu4"],
        WRITE_NOISE_SWEEP["cider"],
        "write_noise_std",
        "write_noise sweep",
        log_x=True,
        bleu4_err=WRITE_NOISE_SWEEP["bleu4_err"],
        cider_err=WRITE_NOISE_SWEEP["cider_err"],
    )
    plot_dual_metric_line(
        axes[1],
        READ_NOISE_SWEEP["std"],
        READ_NOISE_SWEEP["bleu4"],
        READ_NOISE_SWEEP["cider"],
        "read_noise_std",
        "read_noise sweep",
        log_x=True,
        bleu4_err=READ_NOISE_SWEEP["bleu4_err"],
        cider_err=READ_NOISE_SWEEP["cider_err"],
    )
    save_figure(fig, output_dir, "fig6_write_read_noise_sensitivity", formats)


def make_fig7(output_dir: Path, formats: list[str]) -> None:
    """Fig. 7: CMM-only vs CMM->CrossSim gap."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    x = np.arange(len(GAP_DATA))
    width = 0.34
    labels = [row["condition"] for row in GAP_DATA]
    legend_handles = None
    legend_labels = None

    for ax, metric, ylabel, title in [
        (axes[0], "bleu4", "BLEU-4", "BLEU-4 gap"),
        (axes[1], "cider", "CIDEr", "CIDEr gap"),
    ]:
        cmm = [row[f"cmm_{metric}"] for row in GAP_DATA]
        crosssim = [row[f"crosssim_{metric}"] for row in GAP_DATA]
        ax.bar(x - width / 2, cmm, width, label="CMM-only", color="#4c78a8")
        ax.bar(x + width / 2, crosssim, width, label="CMM->CrossSim", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # 共享图例放在底部预留区域，避免遮挡柱状图主体和 x 轴标签。
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
    )

    save_figure(fig, output_dir, "fig7_cmm_only_vs_cmm_crosssim_gap", formats)


def read_spice_validation(csv_path: Path) -> list[dict[str, float | str]]:
    """读取 ngspice 校验 CSV，避免手工复制能耗数值。"""
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [
        {
            "tile_rank": row["tile_rank"],
            "analytical_pj": float(row["representative_vector_analytic_energy_j"]) * 1e12,
            "spice_pj": float(row["spice_integrated_energy_j"]) * 1e12,
            "relative_error_percent": float(row["relative_error"]) * 100.0,
        }
        for row in rows
    ]


def make_fig10(output_dir: Path, formats: list[str], spice_csv: Path) -> None:
    """Fig. 10: analytical vs ngspice tile energy validation."""
    data = read_spice_validation(spice_csv)
    labels = [str(row["tile_rank"]) for row in data]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    ax2 = ax.twinx()
    analytical = [float(row["analytical_pj"]) for row in data]
    spice = [float(row["spice_pj"]) for row in data]
    rel_err = [float(row["relative_error_percent"]) for row in data]

    ax.bar(x - width / 2, analytical, width, label="Analytical energy", color="#4c78a8")
    ax.bar(x + width / 2, spice, width, label="SPICE energy", color="#f58518")
    ax2.plot(x, rel_err, color="#2ca02c", marker="o", linewidth=1.8, label="Relative error")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tile rank")
    ax.set_ylabel("Energy (pJ)")
    ax2.set_ylabel("Relative error (%)", color="#2ca02c")
    ax2.tick_params(axis="y", colors="#2ca02c")
    ax.set_title("Analytical vs ngspice tile energy")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    save_figure(fig, output_dir, "fig10_analytical_vs_ngspice_tile_energy", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated formats, e.g. png,pdf")
    parser.add_argument(
        "--spice-validation-csv",
        type=Path,
        default=Path("experiments/spice/results/cmm_crosssim_read_energy/spice_validation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip().lstrip(".") for item in args.formats.split(",") if item.strip()]

    configure_style()
    make_fig4(output_dir, formats)
    make_fig5(output_dir, formats)
    make_fig6(output_dir, formats)
    make_fig7(output_dir, formats)
    make_fig10(output_dir, formats, args.spice_validation_csv)
    print(f"Saved figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
