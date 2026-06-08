# CrossSim Write Noise 消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-22（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_crosssim_write_noise_conditions.py`；`src/evaluate_crosssim.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估手动写入噪声对 CrossSim `decoder_only` 映射模型的影响，确定在理想 ADC/DAC 条件下模型对编程误差的容忍区间，并找出从“几乎无损”到“明显退化”再到“功能崩溃”的临界点。该实验同时用于回答一个工程问题：在放弃 MemTorch、转向 CrossSim 之后，外部写入误差本身会在什么量级开始主导性能下降。

## 运行命令

### 1. 构建 Write Noise 条件模型

```bash
python -m src.test_crosssim_write_noise_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/crosssim_write_noise_conditions \
  --device cuda \
  --use-gpu \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --adc-resolution 0 \
  --dac-resolution 0 \
  --bias-rows 0 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --read-noise-std 0.0 \
  --programming-error-std 0.0 \
  --seed 42 \
  --save-baseline-crosssim
```

### 2. 对单个条件模型做数值对比评估（示例：it-6）

```bash
python -m src.evaluate_crosssim \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --crosssim-checkpoint checkpoints/crosssim_write_noise_conditions/caption_transformer_it-6_write_noise_crosssim.pt \
  --device cuda \
  --max-batches 100
```

### 3. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/crosssim_write_noise_conditions/conditions_manifest.json \
  --output checkpoints/crosssim_write_noise_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_crosssim_write_noise_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/crosssim_write_noise_conditions | 推断自项目约定/后续文件路径 | checkpoints/crosssim_write_noise_conditions | 条件模型输出目录 |
| --device | cuda | 用户记录 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | True | 用户记录 | False | 显式要求 CrossSim 使用 GPU |
| --scope | decoder_only | 用户记录 | decoder_only | CrossSim 映射范围 |
| --tile-rows | 128 | 用户记录 | 128 | 交叉阵列最大行数 |
| --tile-cols | 128 | 用户记录 | 128 | 交叉阵列最大列数 |
| --adc-resolution | 0 | 用户记录 | 10 | ADC 分辨率；0 表示理想 ADC |
| --dac-resolution | 0 | 用户记录 | 12 | DAC 分辨率；0 表示理想 DAC |
| --bias-rows | 0 | 默认值 | 0 | bias 映射到阵列时使用的额外行数 |
| --rmin | 1000 | 默认值 | 1000.0 | 器件最小电阻 |
| --rmax | 100000 | 默认值 | 100000.0 | 器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续电导 |
| --read-noise-std | 0.0 | 默认值 | 0.0 | 固定读噪声强度；0 表示关闭 |
| --programming-error-std | 0.0 | 默认值 | 0.0 | CrossSim 内部写入误差；0 表示关闭 |
| --seed | 42 | 默认值 | 42 | 随机种子 |
| --save-baseline-crosssim | True | 从结果表推断 | False | 同时保存 noise_std=0 的 CrossSim 基准模型 |
| --conditions | it-10,it-9,it-8,it-7,it-6 | 默认行为 | 空字符串 | 为空时运行全部默认条件 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

### 脚本 2：`src/evaluate_crosssim.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | Baseline training checkpoint |
| --crosssim-checkpoint | 各条件 checkpoint | 用户记录 | 无（required） | CrossSim checkpoint |
| --device | cuda | 用户记录 | `cuda` if available else `cpu` | 运行设备 |
| --batch-size | 16 | 默认值 | 16 | 验证 batch size |
| --num-workers | 0 | 默认值 | 0 | DataLoader worker 数 |
| --max-batches | 100 | 用户记录 | 100 | 最多评估 batch 数 |
| --subset-size | 0 | 默认值 | 0 | 0 表示使用完整验证集 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据根目录 |

### 脚本 3：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/crosssim_write_noise_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/crosssim_write_noise_conditions/metrics_pycoco.json | 用户记录/项目习惯 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### 逐条件数值对比（`evaluate_crosssim.py`, max-batches=100）

| 条件 | 写噪声 std | Baseline Loss | CrossSim Loss | Δ Loss | Baseline Acc | CrossSim Acc | Δ Acc | Logit MAE | Logit Max Error | Logit RMSE |
|------|------------|---------------|---------------|--------|--------------|--------------|-------|-----------|-----------------|------------|
| baseline-crosssim | 0 | 2.534256 | 2.534256 | 0.000000 | 0.476609 | 0.476609 | 0.000000 | 0.000001 | 0.000072 | 0.000001 |
| it-10 | 1e-6 | 2.534256 | 2.534277 | 0.000022 | 0.476609 | 0.476443 | -0.000166 | 0.001048 | 0.046411 | 0.001462 |
| it-9 | 1e-5 | 2.534256 | 2.534284 | 0.000028 | 0.476609 | 0.476277 | -0.000331 | 0.001997 | 0.066716 | 0.002727 |
| it-8 | 1e-4 | 2.534256 | 2.534066 | -0.000190 | 0.476609 | 0.477382 | 0.000773 | 0.012248 | 0.448982 | 0.017217 |
| it-7 | 1e-3 | 2.534256 | 3.308090 | 0.773834 | 0.476609 | 0.392930 | -0.083679 | 1.063812 | 13.412535 | 1.392701 |
| it-6 | 1e-2 | 2.534256 | 5.824668 | 3.290413 | 0.476609 | 0.224745 | -0.251864 | 1.991278 | 21.485548 | 2.520760 |

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|--------|--------|--------|---------|-------|-------|
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.855 | 0.168 |
| baseline-crosssim | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.855 | 0.168 |
| it-10 | 0.6823 | 0.2472 | 0.2280 | 0.4962 | 0.856 | 0.168 |
| it-9 | 0.6823 | 0.2466 | 0.2278 | 0.4955 | 0.855 | 0.168 |
| it-8 | 0.6805 | 0.2502 | 0.2286 | 0.4970 | 0.857 | 0.168 |
| it-7 | 0.4647 | 0.0913 | 0.1171 | 0.3475 | 0.180 | 0.045 |
| it-6 | 0.1561 | 0.0000 | 0.0581 | 0.1993 | 0.008 | 0.001 |

## 备注

- 本实验中的“write noise”是先在 PyTorch 权重上手动注入高斯噪声，再映射为 CrossSim `AnalogLinear`；它不等同于 CrossSim 内部 `programming_error_std` 模型。[src/test_crosssim_write_noise_conditions.py:1-6](src/test_crosssim_write_noise_conditions.py#L1-L6)
- 默认条件集合由脚本内部 `default_noise_settings()` 给出：`it-10=1e-6`、`it-9=1e-5`、`it-8=1e-4`、`it-7=1e-3`、`it-6=1e-2`。[src/test_crosssim_write_noise_conditions.py:91-99](src/test_crosssim_write_noise_conditions.py#L91-L99)
- `it-10 ~ it-8` 基本处于安全区：loss、token acc 与 caption 级指标都与基线几乎一致。
- `it-7` 是明显退化起点：Logit MAE 从 `0.012` 跳升到 `1.064`，BLEU-4 从 `0.2502` 降到 `0.0913`。
- `it-6` 已经功能崩溃：BLEU-4=0，SPICE≈0；日志记录的直接原因是输出长度暴涨（`testlen=10435` vs `reflen=5983`, `ratio=1.74`），生成了大量无效 token。
- `baseline-crosssim` 与 baseline 完全一致，再次验证在零噪声、理想 ADC/DAC 条件下 CrossSim 映射无损。
- 推荐写入精度：`it-8 (1e-4)`。该条件下数值误差仍低（MAE≈`0.012`），caption 级指标与基线持平甚至略优。
- 仍缺失的信息：这组 sweep 的精确执行时间；以及 pycocoevalcap 输出文件是否显式指定了 `--output checkpoints/crosssim_write_noise_conditions/metrics_pycoco.json`，还是之后手动整理到该路径。
