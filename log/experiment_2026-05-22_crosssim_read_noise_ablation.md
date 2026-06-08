# CrossSim Read Noise 消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-22（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_crosssim_read_noise_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估 CrossSim `read_noise_std` 对 `decoder_only` 映射模型 caption 质量的影响，判断在固定 ADC=10 bit、DAC=12 bit 的中等非理想基线下，读噪声是否会像 write noise 一样显著破坏模型性能，还是仅带来可忽略的推理扰动。

## 运行命令

### 1. 构建 Read Noise 条件模型

```bash
python -m src.test_crosssim_read_noise_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/crosssim_read_noise_conditions \
  --device cuda \
  --use-gpu \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --adc-resolution 10 \
  --dac-resolution 12 \
  --bias-rows 0 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --programming-error-std 0.0 \
  --read-noise-stds 0,1e-5,1e-4,1e-3,1e-2
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/crosssim_read_noise_conditions/conditions_manifest.json \
  --output checkpoints/crosssim_read_noise_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_crosssim_read_noise_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/crosssim_read_noise_conditions | 用户记录 | checkpoints/crosssim_read_noise_conditions | 读噪声条件模型输出目录 |
| --device | cuda | 推断自同组 CrossSim 实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | True | 推断自同组 CrossSim 实验环境 | False | 显式要求 CrossSim 使用 GPU |
| --scope | decoder_only | 推断自同组 CrossSim 实验环境 | decoder_only | CrossSim 映射范围 |
| --tile-rows | 128 | 推断自同组 CrossSim 实验环境 | 128 | 交叉阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CrossSim 实验环境 | 128 | 交叉阵列最大列数 |
| --adc-resolution | 10 | 日志分析 + 脚本默认 | 10 | ADC 分辨率 |
| --dac-resolution | 12 | 日志分析 + 脚本默认 | 12 | DAC 分辨率 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射到阵列时使用的额外行数 |
| --rmin | 1000 | 默认值 | 1000.0 | 器件最小电阻 |
| --rmax | 100000 | 默认值 | 100000.0 | 器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续电导 |
| --programming-error-std | 0.0 | 默认值 | 0.0 | CrossSim 内部写入误差；0 表示关闭 |
| --read-noise-stds | 0,1e-5,1e-4,1e-3,1e-2 | 用户记录/默认值 | 0,1e-5,1e-4,1e-3,1e-2 | 扫描的读噪声标准差列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/crosssim_read_noise_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/crosssim_read_noise_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | read_noise_std | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------------|--------|--------|--------|---------|-------|-------|
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| read-noise-0 | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| read-noise-1e-5 | 1e-5 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8548 | 0.1681 |
| read-noise-1e-4 | 1e-4 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8548 | 0.1681 |
| read-noise-1e-3 | 1e-3 | 0.6839 | 0.2506 | 0.2286 | 0.4964 | 0.8597 | 0.1687 |
| read-noise-1e-2 | 1e-2 | 0.6821 | 0.2505 | 0.2277 | 0.4972 | 0.8555 | 0.1663 |

## 备注

- 该脚本不会改动 PyTorch 权重，只通过 CrossSim 的 `read_noise_std` 改变读取阶段噪声强度。[src/test_crosssim_read_noise_conditions.py:1-6](src/test_crosssim_read_noise_conditions.py#L1-L6)
- 默认硬件基线是 `ADC=10 bit`、`DAC=12 bit`，默认读噪声扫描列表是 `0,1e-5,1e-4,1e-3,1e-2`。[src/test_crosssim_read_noise_conditions.py:76-89](src/test_crosssim_read_noise_conditions.py#L76-L89)
- 条件名由脚本自动生成：`0` 对应 `read-noise-0`，非零值使用科学记数法，例如 `read-noise-1e-3`。[src/test_crosssim_read_noise_conditions.py:103-112](src/test_crosssim_read_noise_conditions.py#L103-L112)
- 所有 read noise 条件下的 caption 指标都与 baseline 基本一致，说明读噪声在当前量级内几乎不影响模型功能。
- 与 write noise 对比非常鲜明：`write_noise_std=1e-2` 时模型已崩溃，而 `read_noise_std=1e-2` 时 BLEU-4 仍为 `0.2505`，甚至略高于 baseline 的 `0.2464`。
- `read-noise-1e-3` 和 `read-noise-1e-2` 出现了轻微“正效应”，可能只是统计波动，也可能来自微小扰动带来的隐式正则化，但幅度很小，不足以支撑额外工程设计。
- 结论：read noise 对该 Transformer 图像描述模型基本无害，无需单独设计专门容错机制；主要风险仍来自 write noise 和 DAC 量化。
- 仍缺失的信息：这组 sweep 的精确执行时间；以及是否还做过 `evaluate_crosssim.py` 的 loss / token_acc / logit 级数值评估但未写入当前日志。
