# CMM-CrossSim ADC 分辨率消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-26（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_crosssim_adc_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

在 CMM → CrossSim 两级映射路径上系统评估 ADC 量化精度对 caption 质量的影响，判断将 CMM 参数放入真实 CrossSim 器件仿真后，ADC 敏感性是否比纯 PyTorch CMM 等效模型更强，并找出 CMM-CrossSim 路径下可接受的 ADC 位宽阈值。

## 运行命令

### 1. 构建 CMM-CrossSim ADC 条件模型

```bash
python -m src.test_cmm_crosssim_adc_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_crosssim_adc_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --dac-resolution 0 \
  --read-noise-std 1e-4 \
  --write-noise-std 1e-4 \
  --adc-resolutions 12,10,8,6,4
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_crosssim_adc_conditions/conditions_manifest.json \
  --output checkpoints/cmm_crosssim_adc_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_crosssim_adc_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_crosssim_adc_conditions | 用户记录 | checkpoints/cmm_crosssim_adc_conditions | ADC 条件模型输出目录 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | False | 默认值（用户未显式指定） | False | CrossSim GPU 后端 |
| --scope | decoder_only | 用户记录/默认值 | decoder_only | CMM-on-CrossSim 映射范围 |
| --tile-rows | 128 | 推断自同组实验环境 | 128 | 阵列最大行数 |
| --tile-cols | 128 | 推断自同组实验环境 | 128 | 阵列最大列数 |
| --rmin | 1000 | 推断自同组实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组实验环境 | 100000.0 | Roff，器件最大电阻 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数；0 表示连续 |
| --dac-resolution | 0 | 默认值 | 0 | 固定 DAC 为理想，隔离纯 ADC 效应 |
| --read-noise-std | 1e-4 | 默认值 | 1e-4 | 固定读噪声 |
| --write-noise-std | 1e-4 | 默认值 | 1e-4 | 固定写入噪声 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射额外行数 |
| --adc-resolutions | 12,10,8,6,4 | 用户记录/默认值 | 12,10,8,6,4 | 扫描的 ADC bit 列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本默认在 `write_noise_std=1e-4`、`read_noise_std=1e-4` 的非零基线噪声下运行，不同于纯 CMM ADC 消融的 0 噪声条件。[src/test_cmm_crosssim_adc_conditions.py:1-6](src/test_cmm_crosssim_adc_conditions.py#L1-L6)

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_crosssim_adc_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_crosssim_adc_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | ADC bits | DAC bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------|----------|--------|--------|--------|---------|-------|-------|
| baseline | — | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| adc-12bit | 12 | 0 | 0.6808 | 0.2509 | 0.2292 | 0.4975 | 0.8594 | 0.1683 |
| adc-10bit | 10 | 0 | 0.6640 | 0.2434 | 0.2231 | 0.4845 | 0.8383 | 0.1641 |
| adc-8bit | 8 | 0 | 0.6445 | 0.2166 | 0.2244 | 0.4825 | 0.7542 | 0.1600 |
| adc-6bit | 6 | 0 | 0.2922 | 0.0287 | 0.1168 | 0.2820 | 0.0722 | 0.0634 |
| adc-4bit | 4 | 0 | 0.0615 | ~0 | 0.0132 | 0.0950 | ~0 | 0.0000 |

## 备注

- 本实验默认在 `write_noise_std=1e-4`、`read_noise_std=1e-4` 的非零噪声基线下运行，ADC 效应叠加在轻度读写噪声之上评估，而非纯理想器件条件。[src/test_cmm_crosssim_adc_conditions.py:1-6](src/test_cmm_crosssim_adc_conditions.py#L1-L6)
- `adc-12bit` 基本无损：BLEU-4 0.2509 与 baseline 0.2464 持平甚至略优，CIDEr 0.8594 略高于 baseline。
- `adc-10bit` 已出现可见退化：CIDEr 从 0.8546 降至 0.8383（约 -1.9%），说明 CMM 经 CrossSim 映射后 ADC 敏感性已明显高于纯 PyTorch CMM。
- `adc-8bit` 明显退化：BLEU-4 降至 0.2166，CIDEr 降至 0.7542，caption 质量出现系统性下降。
- `adc-6bit` 接近功能失效：BLEU-4 仅 0.0287，CIDEr 仅 0.0722。
- `adc-4bit` 功能崩溃：BLEU-4≈0，CIDEr≈0，SPICE=0。
- 与纯 CMM ADC 消融的对比鲜明：纯 CMM 中 adc-10/8 仍较稳定、adc-6 仅轻微退化、adc-4 仍保持可用质量；而 CMM-CrossSim 路径下退化曲线显著陡峭，adc-8 已明显下降、adc-6 几乎崩溃。这正向验证了 PyTorch CMM 等效模型确实过于温和，真实 CrossSim 路径放大了 ADC 量化误差。
- 结论：推荐 CMM-CrossSim ADC ≥ 10-bit；12-bit 基本无损，10-bit 轻微退化可接受，8-bit 及以下不建议使用。
- 仍缺失的信息：精确执行时间。
