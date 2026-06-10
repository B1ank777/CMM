# SPICE Crossbar Read Energy Experiments

本目录用于用 ngspice 和解析公式做 circuit-level read energy estimation。

## 共同假设

- 输出列端为理想 0 V 虚地。
- 暂时忽略线阻、寄生电容、selector、sense amplifier 和驱动器内阻。
- 基础公式：`P(t) = VDD * I_total(t)`，`E = integral(P(t) dt)`。
- 主实验能耗范围：`energy_scope=crossbar_core_read_only`。
- 排除项：`adc,dac,sense,digital_bias,layernorm,softmax,residual,encoder`。

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
```

核心口径：

- CMM-CrossSim 和 CrossSim-only 都统计同一 decoder mapped Linear scope。
- 两者使用同一 COCO validation subset、同一 `Vread/pulse_ns`、同一 `activation_scale=per_vector`。
- CrossSim-only 用于回答：加入 CMM 后，相对普通 CrossSim 映射的 read energy 变化是多少。
- 数字 MAC 对照仍是理论参考线：`--digital-mac-energy-pj 1.0` 表示 `1.0 pJ/MAC`。

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

- `summary.csv`：full decoder 与 self-attn/cross-attn/FFN/output projection 分组能耗。
- `tile_detail.csv`：CMM-CrossSim 每个 CrossSim core tile 的电阻、能耗、rank。
- `crosssim_only_tile_detail.csv`：普通 CrossSim-only 对照的每个 tile 能耗。
- `digital_mac_reference.csv`：同一 mapped decoder Linear 的数字 MAC 数和参考能耗。
- `spice_validation.csv`：CMM-CrossSim high/median/low 代表 tile 的解析能量与 ngspice 积分能量对比。
- `spice_validation/`：代表 tile 的 `.cir/.dat/.csv/.log`。

`summary.csv` 中与对照相关的关键字段：

- `energy_j`：CMM-CrossSim read energy。
- `crosssim_only_energy_j`：普通 CrossSim-only read energy。
- `cmm_to_crosssim_only_energy_ratio`：CMM-CrossSim / CrossSim-only。
- `digital_mac_energy_j`：数字 MAC 理论参考能耗。
- `cmm_read_to_digital_mac_energy_ratio`：CMM-CrossSim / 数字 MAC 理论参考。

## SPICE 验证标准

- 默认要求 `relative_error < 1e-4`。
- 若包含 PULSE 上升/下降沿，允许 `relative_error < 1e-3`。
- 每次验证记录 `tran_step_ns`、`pulse_rise_ns`、`pulse_fall_ns`。

## 后续可扩展项

- 行线/列线电阻。
- bitline/wordline 电容。
- selector 或 transistor。
- sense amplifier 输入电阻。
- ADC/DAC/sense/digital peripheral energy。
