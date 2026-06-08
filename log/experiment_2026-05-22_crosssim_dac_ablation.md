# CrossSim DAC 分辨率消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-22（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_crosssim_dac_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估 DAC 量化精度对 CrossSim `decoder_only` 映射模型 caption 质量的影响，在固定 ADC=10 bit 的中等非理想基线下，判断 DAC 是否比 ADC 更容易成为性能瓶颈，并据此给出可接受的 DAC 位宽建议。

## 运行命令

### 1. 构建 DAC 条件模型

```bash
python -m src.test_crosssim_dac_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/crosssim_dac_conditions \
  --device cuda \
  --use-gpu \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --adc-resolution 10 \
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
  --conditions-manifest checkpoints/crosssim_dac_conditions/conditions_manifest.json \
  --output checkpoints/crosssim_dac_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_crosssim_dac_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/crosssim_dac_conditions | 用户记录 | checkpoints/crosssim_dac_conditions | DAC 条件模型输出目录 |
| --device | cuda | 推断自同组 CrossSim 实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | True | 推断自同组 CrossSim 实验环境 | False | 显式要求 CrossSim 使用 GPU |
| --scope | decoder_only | 推断自同组 CrossSim 实验环境 | decoder_only | CrossSim 映射范围 |
| --tile-rows | 128 | 推断自同组 CrossSim 实验环境 | 128 | 交叉阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CrossSim 实验环境 | 128 | 交叉阵列最大列数 |
| --adc-resolution | 10 | 日志分析 + 脚本默认 | 10 | 固定 ADC 分辨率 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射到阵列时使用的额外行数 |
| --rmin | 1000 | 默认值 | 1000.0 | 器件最小电阻 |
| --rmax | 100000 | 默认值 | 100000.0 | 器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续电导 |
| --read-noise-std | 0.0 | 默认值 | 0.0 | 读噪声强度；0 表示关闭 |
| --programming-error-std | 0.0 | 默认值 | 0.0 | CrossSim 内部写入误差；0 表示关闭 |
| --resolutions | 12,10,8,6,4 | 用户记录/默认值 | 12,10,8,6,4 | 扫描的 DAC bit 列表 |
| --save-baseline-crosssim | True | 从结果表推断 | False | 同时保存 DAC=0 的 CrossSim 基准模型 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/crosssim_dac_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/crosssim_dac_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | ADC bits | DAC bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------|----------|--------|--------|--------|---------|-------|-------|
| baseline | — | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| baseline-crosssim | 10 | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| dac-12 | 10 | 12 | 0.5860 | 0.1637 | 0.1772 | 0.4349 | 0.5397 | 0.1242 |
| dac-10 | 10 | 10 | 0.5787 | 0.1602 | 0.1747 | 0.4336 | 0.5373 | 0.1236 |
| dac-8 | 10 | 8 | 0.5759 | 0.1583 | 0.1749 | 0.4338 | 0.5380 | 0.1224 |
| dac-6 | 10 | 6 | 0.5425 | 0.1456 | 0.1673 | 0.4255 | 0.5144 | 0.1187 |
| dac-4 | 10 | 4 | 0.2013 | 0.0411 | 0.0634 | 0.2537 | 0.1639 | 0.0571 |

## 备注

- 该脚本默认只扫描 DAC，ADC 固定为 10 bit；源码头部已明确说明“默认只改变 DAC，ADC 固定在中等非理想基线（10 bit）”。[src/test_crosssim_dac_conditions.py:1-6](src/test_crosssim_dac_conditions.py#L1-L6)
- 默认 DAC 条件集合由 `--resolutions` 给出：`12,10,8,6,4`；若开启 `--save-baseline-crosssim`，还会额外生成 `baseline-crosssim`（DAC=0, ADC=10）。[src/test_crosssim_dac_conditions.py:81-92](src/test_crosssim_dac_conditions.py#L81-L92)
- DAC 的整体影响明显大于 ADC：即使 `dac-12`，BLEU-4 也从 `0.2464` 降到 `0.1637`，而同一时期的 `adc-12` 基本无损。
- `dac-12 ~ dac-8` 形成平台区：BLEU-4 约 `0.16`、CIDEr 约 `0.54`，说明在这个区间单纯继续提高 DAC 位宽收益很小。
- `dac-6` 略差于 `dac-8`，但仍处于同一量级；真正的灾难性退化出现在 `dac-4`。
- `dac-4` 已严重崩溃：BLEU-4 降到 `0.0411`，SPICE 降到 `0.0571`。
- 结论：推荐 `DAC ≥ 6-bit`；其中 `dac-8/10/12` 几乎等价，不必强求更高位宽，但 `dac-4` 不可用。
- 从结果看，DAC 比 ADC 更可能成为系统瓶颈，因为其量化误差更直接地破坏层间信号传递。
- 仍缺失的信息：这组 sweep 的精确执行时间；以及是否还做过额外的 `evaluate_crosssim.py` 数值级对比但未写入当前日志。
