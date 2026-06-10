# CMM-CrossSim Decoder Crossbar Core 读出能耗评估（200张图·GPU加速·SPICE验证）

| 属性 | 值 |
|------|-----|
| **日期** | 2026-06-09 |
| **状态** | 已完成 |
| **脚本** | experiments/spice/estimate_cmm_crosssim_read_energy.py |

## 实验目的

在 COCO 验证集 200 张图像上，统计 CMM-CrossSim decoder 的 crossbar core 读出能耗（仅 I²R 热功耗），并与 CrossSim-only（纯 VTEAM 物理模型，无 CMM 中间层）及数字 MAC 参考值对比。同时通过 ngspice 瞬态仿真验证解析能耗公式的精度。

对比三栏：
- **CMM-CrossSim**：CMM r-state 参数先映射再写入 CrossSim AnalogLinear
- **CrossSim-only**：直接用 CrossSim VTEAM 模型映射（无 CMM 中间层）
- **数字 MAC 参考**：同样 21 个 Linear 层的等价 MAC 数 × 0.9 pJ/MAC

## 运行命令

```powershell
conda run -n mem python experiments/spice/estimate_cmm_crosssim_read_energy.py --limit 200 --batch-size 1 --vread 0.1 --pulse-ns 10 --digital-mac-energy-pj 0.9 --device cuda --use-gpu
```

## 参数说明

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `--limit` | 200 | 用户指定 | 16 | 验证集样本数量 |
| `--batch-size` | 1 | 默认值 | 1 | DataLoader batch size |
| `--vread` | 0.1 | 默认值 | 0.1 | 读电压/V，per-vector 最大绝对值缩放 |
| `--pulse-ns` | 10.0 | 默认值 | 10.0 | 读脉冲宽度/ns |
| `--digital-mac-energy-pj` | 0.9 | 用户指定 | 1.0 | 数字 MAC 能耗参考值 (pJ/MAC)，对应 45nm CMOS |
| `--device` | cuda | 用户指定 | cuda | 推理设备 |
| `--use-gpu` | True | 用户指定 | False | CrossSim GPU 后端加速 |
| `--baseline-checkpoint` | checkpoints/caption_transformer_epoch_10.pt | 默认值 | 同左 | 训练好的数字基线模型 |
| `--cmm-crosssim-checkpoint` | checkpoints/caption_transformer_array-128x128_cmm_crosssim.pt | 默认值 | 同左 | CMM-CrossSim 映射后的 checkpoint |
| `--crosssim-only-checkpoint` | checkpoints/caption_transformer_crosssim_decoder.pt | 默认值 | 同左 | CrossSim-only decoder checkpoint |
| `--output-dir` | experiments/spice/results/cmm_crosssim_read_energy | 默认值 | 同左 | 结果输出目录 |
| `--num-workers` | 0 | 默认值 | 0 | DataLoader worker 数 |
| `--skip-ngspice` | False | 默认值 | False | 未跳过 SPICE 验证 |
| `--spice-pulse-rise-ns` | 0.001 | 默认值 | 0.001 | ngspice PULSE 上升沿 |
| `--spice-pulse-fall-ns` | 0.001 | 默认值 | 0.001 | ngspice PULSE 下降沿 |

## 模型映射信息

| 属性 | 值 |
|------|-----|
| 映射 Linear 层数 | 21 |
| Tile 形状 | 128×128 |
| 总 tile 数 | 532 |
| ADC 分辨率 | 10 bit（来自 CMM-CrossSim checkpoint） |
| DAC 分辨率 | 0（理想，无 DAC 量化） |
| R_on | 1000 Ω |
| R_off | 100000 Ω |
| Read noise std | 0.0001 |
| Write noise std | 0.0001 |

## 实验结果

### 运行统计

| 指标 | 值 |
|------|-----|
| 处理图像数 | 200 |
| 每 tile 平均 activation vector 数 | 9800 |
| 每层 token vector 总数（总和） | 76192 |

### 按组能耗汇总

| 组 | CMM-CrossSim 能耗 (J) | CrossSim-only 能耗 (J) | CMM/CS 能耗比 | 数字 MAC 数 | 数字 MAC 能耗 (J) | CMM读/数字MAC 比 |
|---|---|---|---|---|---|---|
| self_attn | 2.677e-06 | 2.650e-06 | 1.0104 | 1.14e+09 | 1.027e-03 | 0.00261 |
| cross_attn | 1.005e-05 | 1.005e-05 | 1.0007 | 3.14e+09 | 2.825e-03 | 0.00356 |
| ffn | 2.928e-06 | 2.910e-06 | 1.0062 | 2.28e+09 | 2.054e-03 | 0.00143 |
| output_proj | 7.200e-06 | 7.173e-06 | 1.0037 | 4.88e+09 | 4.396e-03 | 0.00164 |
| **full_decoder** | **2.286e-05** | **2.278e-05** | **1.0035** | **1.14e+10** | **1.030e-02** | **0.00222** |

### SPICE 验证结果

| Tile 等级 | Tile ID | 解析能耗 (J) | SPICE 能耗 (J) | 相对误差 | 验证通过 |
|-----------|---------|-------------|---------------|---------|---------|
| low | layers.0.ffn.3\|r0c0\|neg | 1.324e-12 | 1.324e-12 | 6.96e-05 | ✓ |
| median | output_proj\|r35c0\|neg | 1.300e-11 | 1.300e-11 | 6.96e-05 | ✓ |
| high | layers.1.cross_attn.v_proj\|r0c1\|pos | 2.906e-11 | 2.906e-11 | 6.96e-05 | ✓ |

**最大 SPICE 相对误差：6.96e-05**（远低于 1e-4 验证阈值），解析公式与 ngspice 瞬态仿真高度一致。

### 关键发现

1. **CMM vs CrossSim 能耗几乎一致**（ratio = 1.0035）：CMM r-state 模型在电导分布上与 CrossSim VTEAM 物理模型差异极小，说明 CMM 的简化电导映射并未显著改变 crossbar 的阻性功耗特征。
2. **Crossbar 读功耗远低于数字 MAC**（ratio ≈ 0.0022）：crossbar core 的纯 I²R 读功耗仅为等量数字 MAC 能耗的 ~0.2%，体现了模拟域 MVM 的能效优势。但此值不含 ADC/DAC/sense amplifier 等外围电路开销。
3. **cross_attn 组能耗最高**：占 full decoder 的 ~44%，因其 K/V 投影维度最大（QKV 都是 512→512，但 cross_attn 的 memory 侧来自 encoder 的 2048 维特征，激活值更大）。

### 能耗范围说明（ENERGY_SCOPE）

```
crossbar_core_read_only — 仅 crossbar 阵列 I²R 阻性读功耗

排除部分：ADC、DAC、sense amplifier、digital bias、LayerNorm、
         softmax、residual add、encoder（ResNet-50）
```

## 输出文件

| 文件 | 路径 |
|------|------|
| 汇总表 | [summary.csv](../experiments/spice/results/cmm_crosssim_read_energy/summary.csv) |
| Tile 明细 | [tile_detail.csv](../experiments/spice/results/cmm_crosssim_read_energy/tile_detail.csv) |
| CrossSim-only tile 明细 | [crosssim_only_tile_detail.csv](../experiments/spice/results/cmm_crosssim_read_energy/crosssim_only_tile_detail.csv) |
| 数字 MAC 参考 | [digital_mac_reference.csv](../experiments/spice/results/cmm_crosssim_read_energy/digital_mac_reference.csv) |
| SPICE 验证 | [spice_validation.csv](../experiments/spice/results/cmm_crosssim_read_energy/spice_validation.csv) |
| 元数据清单 | [manifest.json](../experiments/spice/results/cmm_crosssim_read_energy/manifest.json) |
| SPICE 网表/波形 | [spice_validation/](../experiments/spice/results/cmm_crosssim_read_energy/spice_validation/) (*.cir, *.csv, *.dat, *.log) |

## 备注

- 运行环境：conda env `mem`，Windows 11，CUDA GPU，CrossSim GPU 后端
- 默认 checkpoint `caption_transformer_array-128x128_cmm_crosssim.pt` 的参数：ADC=10, DAC=0, r_on=1e3, r_off=1e5, read_noise=1e-4, write_noise=1e-4
- 数字 MAC 参考值 0.9 pJ/MAC 取自 Horowitz (ISSCC 2014) 45nm CMOS 8-bit MAC 数据
- 论文引用：Ielmini & Wong (Nature Electronics, 2018) 用于 crossbar read energy 模型；Hu et al. (DAC 2016) 用于 dot-product engine 功耗分析框架
