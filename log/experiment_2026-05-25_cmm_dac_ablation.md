# CMM DAC 分辨率消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_dac_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估论文式 CMM 映射中 DAC 量化精度对 caption 质量的影响。该脚本隔离观测输入 tile DAC 量化的独立效应：ADC 固定为理想 0 bit，cell_bits / write_noise / read_noise 均固定为 0，仅扫描 DAC 分辨率。

## 运行命令

### 1. 构建 CMM DAC 条件模型

```bash
python -m src.test_cmm_dac_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_dac_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --dac-resolutions 0,12,10,8,6,4
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_dac_conditions/conditions_manifest.json \
  --output checkpoints/cmm_dac_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_dac_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_dac_conditions | 用户记录 | checkpoints/cmm_dac_conditions | CMM DAC 条件模型输出目录 |
| --device | cuda | 推断自同组 CMM 实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 推断自同组 CMM 实验环境 | decoder_only | CMM 映射范围 |
| --tile-rows | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 推断自同组 CMM 实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组 CMM 实验环境 | 100000.0 | Roff，器件最大电阻 |
| --dac-resolutions | 0,12,10,8,6,4 | 用户记录/默认值 | 0,12,10,8,6,4 | 扫描的 DAC bit 列表；0 表示理想 DAC |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本内部固定：ADC=0、cell_bits=0、write_noise_std=0.0、read_noise_std=0.0，用于隔离纯 DAC 效应。[src/test_cmm_dac_conditions.py:106-111](src/test_cmm_dac_conditions.py#L106-L111)

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_dac_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_dac_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | ADC bits | DAC bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------|----------|--------|--------|--------|---------|-------|-------|
| baseline | — | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| dac-ideal | 0 | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| dac-12bit | 0 | 12 | 0.6815 | 0.2478 | 0.2278 | 0.4955 | 0.8557 | 0.1682 |
| dac-10bit | 0 | 10 | 0.6835 | 0.2496 | 0.2287 | 0.4963 | 0.8605 | 0.1688 |
| dac-8bit | 0 | 8 | 0.6811 | 0.2462 | 0.2275 | 0.4953 | 0.8523 | 0.1662 |
| dac-6bit | 0 | 6 | 0.6740 | 0.2472 | 0.2275 | 0.4935 | 0.8536 | 0.1684 |
| dac-4bit | 0 | 4 | 0.6702 | 0.2309 | 0.2232 | 0.4846 | 0.8107 | 0.1606 |

## 备注

- 该脚本默认 ADC 固定为 0（理想），cell_bits/write_noise/read_noise 均固定为 0，用于纯粹隔离 DAC 量化的影响。[src/test_cmm_dac_conditions.py:1-7](src/test_cmm_dac_conditions.py#L1-L7)
- DAC 条件名由脚本生成：`0` 对应 `dac-ideal`，其余对应 `dac-{n}bit`。[src/test_cmm_dac_conditions.py:85-88](src/test_cmm_dac_conditions.py#L85-L88)
- `dac-ideal` 与 baseline 完全一致，验证 CMM DAC 量化路径在理想条件下无额外数值误差。
- `dac-12/10/8` 基本无损：各项指标与 baseline 持平，`dac-10` 甚至略优，说明 `≥8-bit` 在 CMM 中 DAC 量化基本无害。
- `dac-6` 仍保持稳定：BLEU-4 0.2472 与 baseline 一致，CIDEr 0.8536 仅下降 0.1%。
- `dac-4` 才开始明显退化：BLEU-4 降至 0.2309（-6.3%），CIDEr 降至 0.8107（-5.1%），但仍远好于 CrossSim dac-4 的崩溃程度。
- 与 CrossSim DAC 的鲜明对比：CrossSim 在 dac-12 就有明显退化，而 CMM 直到 dac-4 才出现可见下降，说明当前 PyTorch CMM 等效模型的 DAC 量化误差传播远弱于真实 CrossSim/VTEAM 硬件路径。
- 结论：推荐 CMM DAC ≥ 6-bit，6-bit 及以上基本无损，8/10/12-bit 无需刻意区分。
- 仍缺失的信息：精确执行时间。
