# CMM ADC 分辨率消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_adc_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估论文式 CMM 映射中 ADC 量化精度对 caption 质量的影响。该脚本隔离观测 tile partial sum ADC 量化的独立效应：DAC 固定为理想 0 bit，cell_bits / write_noise / read_noise 均固定为 0，仅扫描 ADC 分辨率。

## 运行命令

### 1. 构建 CMM ADC 条件模型

```bash
python -m src.test_cmm_adc_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_adc_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --adc-resolutions 0,12,10,8,6,4
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_adc_conditions/conditions_manifest.json \
  --output checkpoints/cmm_adc_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_adc_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_adc_conditions | 用户记录 | checkpoints/cmm_adc_conditions | CMM ADC 条件模型输出目录 |
| --device | cuda | 推断自同组 CMM 实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 推断自同组 CMM 实验环境 | decoder_only | CMM 映射范围 |
| --tile-rows | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 推断自同组 CMM 实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组 CMM 实验环境 | 100000.0 | Roff，器件最大电阻 |
| --adc-resolutions | 0,12,10,8,6,4 | 用户记录/默认值 | 0,12,10,8,6,4 | 扫描的 ADC bit 列表；0 表示理想 ADC |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本内部固定：DAC=0、cell_bits=0、write_noise_std=0.0、read_noise_std=0.0，用于隔离纯 ADC 效应。[src/test_cmm_adc_conditions.py:106-111](src/test_cmm_adc_conditions.py#L106-L111)

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_adc_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_adc_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | ADC bits | DAC bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------|----------|--------|--------|--------|---------|-------|-------|
| baseline | — | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| adc-ideal | 0 | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| adc-12bit | 12 | 0 | 0.6835 | 0.2497 | 0.2288 | 0.4968 | 0.8603 | 0.1691 |
| adc-10bit | 10 | 0 | 0.6851 | 0.2521 | 0.2291 | 0.4980 | 0.8597 | 0.1679 |
| adc-8bit | 8 | 0 | 0.6818 | 0.2491 | 0.2293 | 0.4978 | 0.8517 | 0.1671 |
| adc-6bit | 6 | 0 | 0.6774 | 0.2463 | 0.2258 | 0.4955 | 0.8357 | 0.1655 |
| adc-4bit | 4 | 0 | 0.6539 | 0.2201 | 0.2214 | 0.4756 | 0.7830 | 0.1608 |

## 备注

- 该脚本默认 DAC 固定为 0（理想），cell_bits/write_noise/read_noise 均固定为 0，用于纯粹隔离 ADC 量化的影响。[src/test_cmm_adc_conditions.py:1-7](src/test_cmm_adc_conditions.py#L1-L7)
- ADC 条件名由脚本生成：`0` 对应 `adc-ideal`，其余对应 `adc-{n}bit`。[src/test_cmm_adc_conditions.py:85-88](src/test_cmm_adc_conditions.py#L85-L88)
- `adc-ideal` 与 baseline 完全一致，验证 CMM ADC 量化路径在理想条件下无额外数值误差。
- `adc-12/10/8` 均未出现退化，BLEU-4 与 CIDEr 与 baseline 持平甚至略优，说明 `≥8-bit` 在 CMM 中基本无损。
- `adc-6` 开始轻微退化：CIDEr 从 `0.8546` 降至 `0.8357`（-2.2%），属于边缘可接受区间。
- `adc-4` 明显退化但未崩溃：BLEU-4 下降 10.7%，但仍远好于 CrossSim adc-4 的完全失效。
- 与 CrossSim ADC 的关键差异：CMM ADC 敏感性显著弱于 CrossSim——CrossSim adc-6 已严重退化、adc-4 基本崩溃，而 CMM adc-6 仅轻微退化、adc-4 仍保持可用质量。这说明当前 PyTorch CMM 等效模型的 ADC 量化路径比 CrossSim/VTEAM 模型更平滑、误差传播更弱。[src/test_cmm_adc_conditions.py:4-6](src/test_cmm_adc_conditions.py#L4-L6)
- 结论：推荐 CMM ADC ≥ 6-bit（退化约 2%），8-bit 及以上基本无损。
- 仍缺失的信息：精确执行时间。
