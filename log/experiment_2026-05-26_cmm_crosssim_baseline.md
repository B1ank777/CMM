# CMM-CrossSim 理想基线验证

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-26（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_crosssim_baseline.py`；`src/map_cmm_crosssim.py`；`src/evaluate_crosssim.py` |

## 实验目的

验证 CMM → CrossSim 两级映射链路在理想条件（ADC/DAC=0/0、cell_bits=0、无读写噪声）下的数值正确性。这是继 PyTorch `CMMLinear` 等价性验证之后的关键一步：确认 CMM 参数在写入 CrossSim `AnalogLinear` 后仍可无损复现数字基线。

## 运行命令

### 1. 构建理想 CMM-CrossSim 基线模型

```bash
python -m src.test_cmm_crosssim_baseline \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt
```

### 2. 对理想基线模型做整模数值评估

```bash
python -m src.evaluate_crosssim \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --crosssim-checkpoint checkpoints/cmm_crosssim_conditions/caption_transformer_cmm_crosssim_ideal.pt \
  --device cuda \
  --max-batches 100
```

## 参数说明

### 脚本 1：`src/test_cmm_crosssim_baseline.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output | checkpoints/cmm_crosssim_conditions/caption_transformer_cmm_crosssim_ideal.pt | 用户记录 | checkpoints/cmm_crosssim_conditions/caption_transformer_cmm_crosssim_ideal.pt | 输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 默认值 | decoder_only | 映射范围 |
| --tile-rows | 128 | 默认值 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 默认值 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 默认值 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 默认值 | 100000.0 | Roff，器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续 |
| --adc-resolution | 0 | 默认值 | 0 | ADC 分辨率；0=理想 |
| --dac-resolution | 0 | 默认值 | 0 | DAC 分辨率；0=理想 |
| --write-noise-std | 0.0 | 默认值 | 0.0 | 写入噪声 |
| --read-noise-std | 0.0 | 默认值 | 0.0 | 读噪声 |
| --use-gpu | False | 从评估日志 CrossSim GPU: False 推断 | False | CrossSim GPU 后端 |

### 脚本 2：`src/evaluate_crosssim.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | Baseline checkpoint |
| --crosssim-checkpoint | checkpoints/cmm_crosssim_conditions/caption_transformer_cmm_crosssim_ideal.pt | 用户记录 | 无（required） | CMM-CrossSim checkpoint |
| --device | cuda | 用户记录 | `cuda` if available else `cpu` | 运行设备 |
| --batch-size | 16 | 默认值 | 16 | 验证 batch size |
| --num-workers | 0 | 默认值 | 0 | DataLoader worker 数 |
| --max-batches | 100 | 用户记录 | 100 | 最多评估 batch 数 |
| --subset-size | 0 | 默认值 | 0 | 0 表示使用完整验证集 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据根目录 |

## 实验结果

### 整模数值对比（`evaluate_crosssim.py`, max-batches=100）

| 指标 | Baseline | CMM-CrossSim | 差值 |
|------|----------|-------------|------|
| loss | 2.534256 | 2.534256 | 0.000000 |
| token_acc | 0.476609 | 0.476609 | 0.000000 |
| Logit MAE | — | 0.000001 | — |
| Logit Max Error | — | 0.000072 | — |
| Logit RMSE | — | 0.000001 | — |

**映射元数据**：
- Format: `cmm_crosssim_v1`
- Mapped linear layers: 21
- Scope: `decoder_only`
- Tile shape: (128, 128)
- Rmin/Rmax: 1000.0 / 100000.0
- Cell bits: 0
- Write/Read noise: 0.0 / 0.0
- ADC/DAC: 0 / 0

## 备注

- 该实验是两阶段链路的关键验证节点：CMM 参数先被映射为 `r_pos/r_neg` 内部状态，再通过 `CMMLinear.from_linear()` 写入 CrossSim `AnalogLinear`。[src/map_cmm_crosssim.py](src/map_cmm_crosssim.py)
- 理想条件下（ADC/DAC=0、cell_bits=0、无噪声），CMM-CrossSim 的 loss / token_acc 与 baseline 完全一致，Logit MAE 仅为 `1e-6` 浮点舍入量级。
- 21 个线性层全部成功映射，checkpoint 元数据完整可读，包含 `crosssim_args` 和 `cmm_crosssim_args` 双参数组。
- 与前面 2026-05-25 的纯 PyTorch `CMMLinear` 等价性验证构成两级基线：前者验证 CMM 等效模型本身正确，这里进一步验证 CMM → CrossSim 映射链路正确。
- 结论：`cmm_crosssim_v1` 映射链路在理想条件下数值正确，可作为后续非理想消融实验的基线。
- 仍缺失的信息：精确执行时间；`test_cmm_crosssim_baseline.py` 的完整命令行参数是否完全使用默认值，还是显式指定了部分参数。
