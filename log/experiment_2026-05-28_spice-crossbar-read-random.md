# SPICE Crossbar Read 仿真 - 低占空比 random 模式

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-28 |
| **状态** | 已完成 |
| **脚本** | experiments/spice/run_crossbar_read.py |

## 实验目的

建立 SPICE 仿真基线，验证 crossbar read 仿真流程，测量低占空比 random 模式（on_prob=0.3）下的读出功耗和列电流分布。

## 运行命令

```bash
python experiments/spice/run_crossbar_read.py --n 16 --vread 0.1 --pulse-ns 10 --ron 5000 --roff 500000 --pattern random --on-prob 0.3
```

## 参数说明

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --n | 16 | 用户指定 | 16 | crossbar 阵列规模 |
| --vread | 0.1 | 用户指定 | 0.1 | read pulse 电压 (V) |
| --pulse-ns | 10 | 用户指定 | 10.0 | read pulse 宽度 (ns) |
| --ron | 5000 | 用户指定 | — | Ron 低阻态电阻 (Ohm) |
| --roff | 500000 | 用户指定 | — | Roff 高阻态电阻 (Ohm) |
| --pattern | random | 用户指定 | checker | Ron/Roff 空间分布模式 |
| --on-prob | 0.3 | 用户指定 | 0.5 | random 模式下的 Ron 概率 |
| --seed | 1 | 默认值 | 1 | random 模式随机种子 |

## 实验结果

### 输出文件

| 类型 | 路径 |
|------|------|
| Netlist | `experiments/spice/results/n16_v0.1_pw10ns_ron5k_roff500k_random_on0.3_seed1.cir` |
| Waveform (raw) | `experiments/spice/results/n16_v0.1_pw10ns_ron5k_roff500k_random_on0.3_seed1.dat` |
| Waveform (csv) | `experiments/spice/results/n16_v0.1_pw10ns_ron5k_roff500k_random_on0.3_seed1.csv` |
| SPICE Log | `experiments/spice/results/n16_v0.1_pw10ns_ron5k_roff500k_random_on0.3_seed1.log` |

### 功耗

| 指标 | 值 |
|------|-----|
| 总能量 | 1.585 pJ |
| 平均功率 | 158.5 µW |
| 峰值功率 | 159.6 µW |

### 稳态电流

| 指标 | 值 |
|------|-----|
| 行总电流 | 1.596 mA |
| 列总电流 | 1.596 mA |
| 列最小电流 | 42.8 µA |
| 列最大电流 | 181.4 µA |
| 列动态范围 | 4.24× |

## 备注

- 16×16 crossbar，Ron/Roff = 5k/500k (on/off ratio = 100×)
- Random 模式下仅 30% 单元为低阻态（on_prob=0.3），属于稀疏接通场景
- 行/列总电流一致（1.596 mA），符合 KCL；列电流动态范围 4.24× 反映了 random 分布的不均匀性
- 默认 seed=1，后续若需多 seed 统计需显式指定 `--seed`
