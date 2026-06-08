# CrossSim Array Size 消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-22（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_crosssim_array_size_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估 CrossSim 交叉阵列 tile 尺寸对 `decoder_only` 映射模型 caption 质量的影响，验证在固定 ADC=10 bit、DAC=12 bit 的中等非理想基线下，`64×64` 到 `512×512` 的阵列切分是否会引入额外精度损失，并检查 CrossSim 的 tile 切分/重组机制是否正确。

## 运行命令

### 1. 构建 Array Size 条件模型

```bash
python -m src.test_crosssim_array_size_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/crosssim_array_size_conditions \
  --device cuda \
  --use-gpu \
  --scope decoder_only \
  --adc-resolution 10 \
  --dac-resolution 12 \
  --bias-rows 0 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --read-noise-std 0.0 \
  --programming-error-std 0.0 \
  --array-sizes 64,128,256,512
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/crosssim_array_size_conditions/conditions_manifest.json \
  --output checkpoints/crosssim_array_size_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_crosssim_array_size_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/crosssim_array_size_conditions | 用户记录 | checkpoints/crosssim_array_size_conditions | array size 条件模型输出目录 |
| --device | cuda | 推断自同组 CrossSim 实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | True | 推断自同组 CrossSim 实验环境 | False | 显式要求 CrossSim 使用 GPU |
| --scope | decoder_only | 推断自同组 CrossSim 实验环境 | decoder_only | CrossSim 映射范围 |
| --adc-resolution | 10 | 日志分析 + 脚本默认 | 10 | ADC 分辨率 |
| --dac-resolution | 12 | 日志分析 + 脚本默认 | 12 | DAC 分辨率 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射到阵列时使用的额外行数 |
| --rmin | 1000 | 默认值 | 1000.0 | 器件最小电阻 |
| --rmax | 100000 | 默认值 | 100000.0 | 器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续电导 |
| --read-noise-std | 0.0 | 默认值 | 0.0 | 读噪声强度；0 表示关闭 |
| --programming-error-std | 0.0 | 默认值 | 0.0 | CrossSim 内部写入误差；0 表示关闭 |
| --array-sizes | 64,128,256,512 | 用户记录/默认值 | 64,128,256,512 | 扫描的方阵 tile 尺寸列表 |
| --rect-array-sizes | 空 | 默认值 | 空字符串 | 若提供矩形尺寸则覆盖 `--array-sizes` |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/crosssim_array_size_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/crosssim_array_size_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | tile_shape | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|------------|--------|--------|--------|---------|-------|-------|
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-64×64 | 64×64 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-128×128 | 128×128 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-256×256 | 256×256 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-512×512 | 512×512 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |

## 备注

- 该脚本默认只改变 tile 的最大行列数，ADC/DAC 仍使用中等非理想基线；源码头部说明了如需验证纯理想 tile 等价性，应显式传入 `--adc-resolution 0 --dac-resolution 0`。[src/test_crosssim_array_size_conditions.py:1-7](src/test_crosssim_array_size_conditions.py#L1-L7)
- 默认扫描的是方阵尺寸 `64,128,256,512`；也支持通过 `--rect-array-sizes` 传入矩形 tile，但本次实验未使用。[src/test_crosssim_array_size_conditions.py:81-93](src/test_crosssim_array_size_conditions.py#L81-L93)
- 条件名由脚本自动生成，格式为 `array-{rows}x{cols}`。[src/test_crosssim_array_size_conditions.py:119-122](src/test_crosssim_array_size_conditions.py#L119-L122)
- `64×64 ~ 512×512` 全部与 baseline 完全一致，说明在当前条件下 tile 切分不会额外降低 caption 质量。
- 这也从侧面验证了 CrossSim 的 tile 切分与重组实现是正确的，没有引入实现层面的数值误差。
- 单独的 array size sweep 无法暴露问题，其真正影响更可能体现在与非理想效应的交互上：更小 tile 会带来更多边界、更多 ADC/DAC 调用以及更多噪声注入点。
- 结论：在当前实验设置下，array size 不是主导误差来源；若要研究阵列尺寸影响，后续应与 write noise、ADC/DAC 量化等条件联合消融。
- 仍缺失的信息：这组 sweep 的精确执行时间；以及是否还做过理想 ADC/DAC（0/0）下的纯 tile 等价性专门验证但未写入当前日志。
