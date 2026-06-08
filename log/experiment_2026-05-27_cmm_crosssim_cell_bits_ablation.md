# CMM-CrossSim cell_bits 量化消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-27（模型构建）/ 2026-05-28（指标评测，具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_crosssim_cell_bits_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

在 CMM → CrossSim 两级映射路径上系统评估 CMM `cell_bits` 量化精度对 caption 质量的影响。`cell_bits` 控制 CMM `r_pos`/`r_neg` 内部状态的离散化位数，0 表示连续电导。该实验验证在 CrossSim 真实器件仿真环境下，CMM 的 r-state 量化是否仍然像纯 PyTorch CMM 中那样基本无损。

## 运行命令

### 1. 构建 CMM-CrossSim cell_bits 条件模型

```bash
python -m src.test_cmm_crosssim_cell_bits_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_crosssim_cell_bits_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --adc-resolution 0 \
  --dac-resolution 0 \
  --read-noise-std 1e-4 \
  --write-noise-std 1e-4 \
  --cell-bits-list 0,2,3,4,6,8
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_crosssim_cell_bits_conditions/conditions_manifest.json \
  --output checkpoints/cmm_crosssim_cell_bits_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_crosssim_cell_bits_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_crosssim_cell_bits_conditions | 默认值 | checkpoints/cmm_crosssim_cell_bits_conditions | 条件模型输出目录 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | False | 默认值 | False | CrossSim GPU 后端 |
| --scope | decoder_only | 默认值 | decoder_only | CMM-on-CrossSim 映射范围 |
| --tile-rows | 128 | 默认值 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 默认值 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 默认值 | 1000.0 | Ron |
| --rmax | 100000 | 默认值 | 100000.0 | Roff |
| --adc-resolution | 0 | 默认值 | 0 | 固定 ADC 为理想 |
| --dac-resolution | 0 | 默认值 | 0 | 固定 DAC 为理想 |
| --read-noise-std | 1e-4 | 默认值 | 1e-4 | 固定读噪声 |
| --write-noise-std | 1e-4 | 默认值 | 1e-4 | 固定写入噪声 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射额外行数 |
| --seed | 42 | 默认值 | 42 | 所有条件共用的随机种子 |
| --cell-bits-list | 0,2,3,4,6,8 | 默认值 | 0,2,3,4,6,8 | 扫描的 cell_bits 列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本内部硬固定：`tile_rows=128`、`tile_cols=128`、`adc_resolution=0`、`dac_resolution=0`、`read_noise_std=1e-4`、`write_noise_std=1e-4`，确保实验变量仅为 `cell_bits`。

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_crosssim_cell_bits_conditions/conditions_manifest.json | 用户指定 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_crosssim_cell_bits_conditions/metrics_pycoco.json | 用户指定 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户指定 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | cell_bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|-----------|--------|--------|--------|---------|-------|-------|
| baseline | — | 0.6821 | 0.2481 | 0.2279 | 0.4957 | 0.8560 | 0.1682 |
| cell-continuous | 0 | 0.6807 | 0.2471 | 0.2273 | 0.4950 | 0.8520 | 0.1669 |
| cell-2bit | 2 | 0.0121 | ~0 | 0.0140 | 0.0172 | 0.0002 | 0.0143 |
| cell-3bit | 3 | 0.5613 | 0.1710 | 0.1976 | 0.4238 | 0.6621 | 0.1416 |
| cell-4bit | 4 | 0.6526 | 0.2262 | 0.2243 | 0.4890 | 0.8010 | 0.1619 |
| cell-6bit | 6 | 0.6720 | 0.2449 | 0.2253 | 0.4897 | 0.8388 | 0.1650 |
| cell-8bit | 8 | 0.6785 | 0.2455 | 0.2277 | 0.4945 | 0.8504 | 0.1665 |

## 备注

- 本实验在 `write_noise_std=1e-4`、`read_noise_std=1e-4`、ADC/DAC=0/0 的轻度非理想基线下运行，`cell_bits` 作为唯一扫描变量。
- `cell-continuous` (cell_bits=0) 与 baseline 几乎一致，确认 CMM-CrossSim 映射在连续电导条件下数值正确。
- **cell-2bit 功能完全崩溃**：BLEU-4≈0、CIDEr≈0.0002、SPICE=0.014。这与纯 PyTorch CMM 的 cell_bits 消融形成鲜明对比——纯 CMM 中 2-bit ~ continuous，而 CMM-CrossSim 路径下 2-bit 量化导致模型输出完全退化。原因：CMM 的 r-state 量化误差经 CrossSim VTEAM 电阻映射后被进一步放大。
- **cell-3bit 已有显著恢复**：BLEU-4 回升至 0.1710、CIDEr 回升至 0.6621（约 baseline 的 77%），但仍远低于可接受水平。
- **cell-4bit**：BLEU-4 0.2262（baseline 的 91%）、CIDEr 0.8010（baseline 的 94%），已有实用价值但仍可见退化。
- **cell-6bit**：BLEU-4 0.2449（baseline 的 99%）、CIDEr 0.8388（baseline 的 98%），退化轻微。
- **cell-8bit**：BLEU-4 0.2455、CIDEr 0.8504，已非常接近 baseline 和 cell-continuous 水平。
- 跨模型对比：
  - 纯 PyTorch CMM：cell-2bit ~ continuous，量化基本无损
  - CMM-CrossSim：cell-2bit 功能崩溃，cell-3bit 严重退化，需 ≥ 6-bit 才接近 baseline
  - 这进一步支持核心判断：PyTorch CMM 等效模型过于温和，真实 CrossSim 路径会放大 CMM r-state 量化误差
- 结论：CMM-CrossSim 路径下推荐 cell_bits ≥ 6-bit；8-bit 基本无损；4-bit 以下不建议使用；2-bit 功能完全失效。
- 仍缺失的信息：精确执行时间。
