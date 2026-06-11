# SPICE Crossbar Read Energy Experiments

本目录用于用 ngspice 和解析公式做 circuit-level read energy estimation。

## 共同假设

- 输出列端为理想 0 V 虚地。
- 暂时忽略线阻、寄生电容、selector、sense amplifier 和驱动器内阻。
- 基础公式：`P(t) = VDD * I_total(t)`，`E = integral(P(t) dt)`。
- 主实验能耗范围：`energy_scope=peripheral_aware_read_energy`。
- 主能耗公式：`energy_j = core_read_energy_j + adc_energy_j + dac_energy_j + digital_accum_energy_j + bias_energy_j`。
- 仍未计入项：`sense,layernorm,softmax,residual,encoder`。

## 实验 A：基础 Crossbar 读功耗

运行示例：

```powershell
python experiments/spice/run_crossbar_read.py --n 16 --vread 0.1 --pulse-ns 10
python experiments/spice/run_crossbar_read.py --n 32 --vread 0.2 --pulse-ns 100 --pattern checker
```

## 实验 B：Ron/Roff 和 Cell State 对功耗的影响

运行示例：

```powershell
python experiments/spice/sweep_ron_roff_cell_state.py
python experiments/spice/sweep_ron_roff_cell_state.py --n 32 --vread 0.2 --pulse-ns 50
```

扫描条件：

- Ron/Roff：`1k/100k`、`5k/500k`、`10k/1M`
- 高电导 Ron cell 占比：`10%`、`30%`、`50%`

## 实验 C：CMM-CrossSim Activation-Aware Read Energy

默认 checkpoint：

- baseline：`checkpoints/caption_transformer_epoch_10.pt`
- CMM-CrossSim：`checkpoints/caption_transformer_array-128x128_cmm_crosssim.pt`
- CrossSim-only：`checkpoints/caption_transformer_crosssim_decoder.pt`

运行示例：

```powershell
conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 1 --batch-size 1 --device cpu --skip-ngspice
conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 16 --batch-size 1 --device cuda --use-gpu
conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 1 --batch-size 1 --device cpu --skip-ngspice --adc-energy-pj-per-conv 1.0 --dac-energy-pj-per-conv 0.1 --digital-mac-energy-pj 0.9 --peripheral-energy-source "parameterized_literature_model"
```

核心口径：

- CMM-CrossSim 和 CrossSim-only 都统计同一 decoder mapped Linear scope。
- 两者使用同一 COCO validation subset、同一 `Vread/pulse_ns`、同一 `activation_scale=per_vector`。
- CMM-CrossSim 和 CrossSim-only 使用同一 ADC/DAC/digital/bias 参数化外围能耗模型，保证能耗对比口径一致。
- CrossSim-only 用于回答：加入 CMM 后，相对普通 CrossSim 映射的 peripheral-aware energy 变化是多少。
- 数字 MAC 对照使用同一批 decoder mapped Linear 的 MAC 数：`--digital-mac-energy-pj 1.0` 表示 `1.0 pJ/MAC`，不包含 encoder、LayerNorm、softmax、residual。
- `--adc-energy-pj-per-conv` 和 `--dac-energy-pj-per-conv` 默认为 0；论文正式数值应显式传入，并用 `--peripheral-energy-source` 记录来源。

CrossSim core 访问路径：

```text
OFFSET:
layer.core.core.cores[r][c].core.matrix

BALANCED:
layer.core.core.cores[r][c].core_pos.matrix
layer.core.core.cores[r][c].core_neg.matrix
```

电阻转换：

```text
G_physical = G_raw * (1 / R_on - 1 / R_off) + 1 / R_off
R_physical = 1 / G_physical
```

注意：CrossSim `matrix` 方向是 `weight[out, in]`，SPICE 行电压对应 input feature，因此代表 tile 网表使用 `R_physical.T`。

## 输出文件

输出目录：

```text
experiments/spice/results/cmm_crosssim_read_energy/
```

主要输出：

- `summary.csv`：full decoder 与 self-attn/cross-attn/FFN/output projection 分组 peripheral-aware 能耗。
- `tile_detail.csv`：CMM-CrossSim 每个 CrossSim core tile 的电阻、core/ADC/DAC/digital/bias 分项、rank。
- `crosssim_only_tile_detail.csv`：普通 CrossSim-only 对照的每个 tile 分项能耗。
- `digital_mac_reference.csv`：同一 mapped decoder Linear 的数字 MAC 数和参考能耗。
- `spice_validation.csv`：CMM-CrossSim high/median/low 代表 tile 的 core read 解析能量与 ngspice 积分能量对比。
- `spice_validation/`：代表 tile 的 `.cir/.dat/.csv/.log`。

`summary.csv` 中与对照相关的关键字段：

- `energy_j`：CMM-CrossSim peripheral-aware total energy。
- `core_read_energy_j` / `adc_energy_j` / `dac_energy_j` / `digital_accum_energy_j` / `bias_energy_j`：CMM-CrossSim 分项能耗。
- `crosssim_only_energy_j`：普通 CrossSim-only peripheral-aware total energy。
- `crosssim_only_core_read_energy_j` / `crosssim_only_adc_energy_j` / `crosssim_only_dac_energy_j` / `crosssim_only_digital_accum_energy_j` / `crosssim_only_bias_energy_j`：CrossSim-only 分项能耗。
- `cmm_to_crosssim_only_energy_ratio`：CMM-CrossSim / CrossSim-only。
- `digital_mac_energy_j`：数字 MAC 理论参考能耗。
- `cmm_to_digital_mac_energy_ratio`：CMM-CrossSim / 数字 MAC 理论参考。
- `crosssim_only_to_digital_mac_energy_ratio`：CrossSim-only / 数字 MAC 理论参考。

## SPICE 验证标准

- 默认要求 `relative_error < 1e-4`。
- 若包含 PULSE 上升/下降沿，允许 `relative_error < 1e-3`。
- 每次验证记录 `tran_step_ns`、`pulse_rise_ns`、`pulse_fall_ns`。
- SPICE 仅验证 crossbar core I²R 子项的解析公式和网表生成；最终 `energy_j` 还包含参数化 ADC/DAC 和可选数字项。

## 后续可扩展项

- 行线/列线电阻。
- bitline/wordline 电容。
- selector 或 transistor。
- sense amplifier 输入电阻。
- sense amplifier 和驱动器能耗。
