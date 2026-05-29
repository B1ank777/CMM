# SPICE crossbar 读功耗实验

本目录用于用 ngspice 做 circuit-level read energy estimation。

## 共同实验假设

- 输出列接 0V 理想虚地。
- 暂时忽略线阻、寄生电容、selector、sense amplifier 和驱动器内阻。
- 基础公式：`P(t) = VDD * I_total(t)`，`E = integral(P(t) dt)`。

## 实验 A：基础 crossbar 读功耗

运行示例：

```powershell
python experiments/spice/run_crossbar_read.py --n 16 --vread 0.1 --pulse-ns 10
python experiments/spice/run_crossbar_read.py --n 32 --vread 0.2 --pulse-ns 100 --pattern checker
```

## 实验 B：Ron/Roff 和 cell state 对功耗的影响

运行示例：

```powershell
python experiments/spice/sweep_ron_roff_cell_state.py
python experiments/spice/sweep_ron_roff_cell_state.py --n 32 --vread 0.2 --pulse-ns 50
```

实验 B 扫描：

- Ron/Roff：`1k/100k`、`5k/500k`、`10k/1M`
- 高电导 Ron cell 占比：`10%`、`30%`、`50%`

## 实验 C：CMM-CrossSim activation-aware read energy

该实验使用 CMM-CrossSim checkpoint 中的真实 CrossSim core conductance。OFFSET core 使用：

```text
layer.core.core.cores[r][c].core.matrix
```

BALANCED core 使用正/负两套阵列，并分别计入能耗：

```text
layer.core.core.cores[r][c].core_pos.matrix
layer.core.core.cores[r][c].core_neg.matrix
```

并结合 COCO 验证集真实 activation 估计 decoder CMM crossbar core read energy。

运行示例：

```powershell
conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 1 --batch-size 1 --device cpu --skip-ngspice
conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 16 --batch-size 1 --device cuda --use-gpu
```

默认输入：

- baseline：`checkpoints/caption_transformer_epoch_10.pt`
- CMM-CrossSim：`checkpoints/caption_transformer_array-128x128_cmm_crosssim.pt`
- `Vread = 0.1 V`
- `pulse_width = 10 ns`
- 数字 MAC 参考：`--digital-mac-energy-pj 1.0`，表示 `1.0 pJ/MAC` 的可配置参考线

核心 metadata：

```text
activation_scale = per_vector
energy_scope = crossbar_core_read_only
excluded = adc,dac,sense,digital_bias,layernorm,softmax,residual,encoder
digital_reference_scope = same_decoder_mapped_linear_mac_only
digital_mac_model = configurable_reference
```

电阻转换：

```text
G_physical = G_raw * (1 / R_on - 1 / R_off) + 1 / R_off
R_physical = 1 / G_physical
```

注意：CrossSim `matrix` 方向是 `weight[out, in]`，SPICE 网表里行电压对应 input feature，因此生成代表 tile 网表时会使用 `R_physical.T`。

high / median / low tile 选择：

- `high`：统计窗口内 `tile_total_energy` 最大。
- `median`：非零 `tile_total_energy` 的中位数。
- `low`：非零 `tile_total_energy` 最小。

代表性 activation vector 选择：

- 对被选中的 tile，选择单次功耗最接近该 tile 平均单次功耗的 vector。
- ngspice 验证的是该具体 vector 的能量；`tile_total_energy` 只用于选择 tile 类型。

输出目录：

```text
experiments/spice/results/cmm_crosssim_read_energy/
```

主要输出：

- `summary.csv`：full decoder 与 self-attn/cross-attn/FFN/output projection 分组能耗。
- `digital_mac_reference.csv`：同一 mapped decoder Linear 在同一 activation 统计窗口下的数字 MAC 数和参考能耗。
- `tile_detail.csv`：每个 CrossSim core tile 的电阻统计、累计能耗和 rank。
- `spice_validation.csv`：high/median/low 代表 tile 的解析能量与 ngspice 积分能量对比。
- `spice_validation/`：代表 tile 的 `.cir/.dat/.csv/.log`。

SPICE 验证误差标准：

- 默认要求 `relative_error < 1e-4`。
- 若使用包含 PULSE 上升/下降沿的默认瞬态设置，允许 `relative_error < 1e-3`。
- 每次验证记录 `tran_step_ns`、`pulse_rise_ns`、`pulse_fall_ns`。
- CMM-CrossSim 代表 tile 验证默认使用 `--spice-pulse-rise-ns 0.001` 和 `--spice-pulse-fall-ns 0.001`，让 SPICE 更接近解析公式的理想方波口径。

## 输出文件说明

`run_crossbar_read.py` 和 CMM-CrossSim 验证会生成：

- `.cir`：ngspice 网表。
- `.dat`：ngspice 原始瞬态数据。
- `.csv`：带表头的波形数据，方便导入 pandas、Excel 或画图工具。
- `.log`：ngspice 输出日志。

## 后续可扩展项

- 行线/列线电阻。
- bitline 电容和 wordline 电容。
- selector 或 transistor。
- sense amplifier 输入电阻。
- ADC/DAC/sense/digital peripheral energy。
