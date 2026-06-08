# CrossSim ADC 分辨率消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-22（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_crosssim_adc_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估 ADC 量化精度对 CrossSim `decoder_only` 映射模型 caption 质量的影响，在固定 DAC=12 bit 的中等非理想基线下，找出模型从“几乎无损”到“明显退化”再到“基本失效”的 ADC 分辨率阈值，并据此给出可接受的 ADC 位宽建议。

## 运行命令

### 1. 构建 ADC 条件模型

```bash
python -m src.test_crosssim_adc_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/crosssim_adc_conditions \
  --device cuda \
  --use-gpu \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --dac-resolution 12 \
  --bias-rows 0 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --read-noise-std 0.0 \
  --programming-error-std 0.0 \
  --resolutions 12,10,8,6,4 \
  --save-baseline-crosssim
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/crosssim_adc_conditions/conditions_manifest.json \
  --output checkpoints/crosssim_adc_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_crosssim_adc_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/crosssim_adc_conditions | 用户记录 | checkpoints/crosssim_adc_conditions | ADC 条件模型输出目录 |
| --device | cuda | 推断自同组 CrossSim 实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | True | 推断自同组 CrossSim 实验环境 | False | 显式要求 CrossSim 使用 GPU |
| --scope | decoder_only | 推断自同组 CrossSim 实验环境 | decoder_only | CrossSim 映射范围 |
| --tile-rows | 128 | 推断自同组 CrossSim 实验环境 | 128 | 交叉阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CrossSim 实验环境 | 128 | 交叉阵列最大列数 |
| --dac-resolution | 12 | 脚本默认 + 日志分析 | 12 | 固定 DAC 分辨率 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射到阵列时使用的额外行数 |
| --rmin | 1000 | 默认值 | 1000.0 | 器件最小电阻 |
| --rmax | 100000 | 默认值 | 100000.0 | 器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续电导 |
| --read-noise-std | 0.0 | 默认值 | 0.0 | 读噪声强度；0 表示关闭 |
| --programming-error-std | 0.0 | 默认值 | 0.0 | CrossSim 内部写入误差；0 表示关闭 |
| --resolutions | 12,10,8,6,4 | 用户记录/默认值 | 12,10,8,6,4 | 扫描的 ADC bit 列表 |
| --save-baseline-crosssim | True | 从结果表推断 | False | 同时保存 ADC=0 的 CrossSim 基准模型 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/crosssim_adc_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/crosssim_adc_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | ADC bits | DAC bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------|----------|--------|--------|--------|---------|-------|-------|
| baseline | — | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| baseline-crosssim | 0 | 12 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| adc-12 | 12 | 12 | 0.6770 | 0.2468 | 0.2282 | 0.4953 | 0.8552 | 0.1680 |
| adc-10 | 10 | 12 | 0.6638 | 0.2396 | 0.2238 | 0.4893 | 0.8284 | 0.1649 |
| adc-8 | 8 | 12 | 0.6310 | 0.2166 | 0.2204 | 0.4785 | 0.7429 | 0.1561 |
| adc-6 | 6 | 12 | 0.2845 | 0.0265 | 0.1167 | 0.2785 | 0.0612 | 0.0659 |
| adc-4 | 4 | 12 | 0.0617 | ~0 | 0.0131 | 0.0951 | ~0 | 0.0 |

## 备注

- 该脚本默认只扫描 ADC，DAC 固定为 12 bit；源码头部已明确说明“默认只改变 ADC，DAC 固定在中等非理想基线（12 bit）”。[src/test_crosssim_adc_conditions.py:1-6](src/test_crosssim_adc_conditions.py#L1-L6)
- 默认 ADC 条件集合由 `--resolutions` 给出：`12,10,8,6,4`；若开启 `--save-baseline-crosssim`，还会额外生成 `baseline-crosssim`（ADC=0, DAC=12）。[src/test_crosssim_adc_conditions.py:81-92](src/test_crosssim_adc_conditions.py#L81-L92)
- `adc-12` 几乎无损：所有 caption 指标与 baseline 基本一致，说明在 12 bit ADC 下量化误差尚未显著影响模型行为。
- `adc-10` 为轻微退化区：CIDEr 从 `0.8546` 降到 `0.8284`，BLEU-4 从 `0.2464` 降到 `0.2396`，退化约 3%，仍可接受。
- `adc-8` 开始明显退化：CIDEr 下降到 `0.7429`，BLEU-4 下降到 `0.2166`。
- `adc-6` 已接近失效：BLEU-4 仅 `0.0265`，CIDEr 仅 `0.0612`。
- `adc-4` 功能崩溃：BLEU-4≈0，SPICE=0，CIDEr≈0。
- 结论：推荐 `ADC ≥ 10-bit`。其中 `adc-10` 退化可控，而 `adc-8` 已明显下降，`adc-6` 及以下不可用。
- 仍缺失的信息：这组 sweep 的精确执行时间；是否还做过 `evaluate_crosssim.py` 层面的 loss / token_acc / logit MAE 数值对比但未写入当前日志。
