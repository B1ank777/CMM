# CMM Array Size 消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_array_size_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估论文式 CMM 映射中 tile 切分尺寸对 caption 质量的影响。在 cell_bits/write_noise/read_noise 均为 0 的理想条件下，扫描 64×64 ~ 512×512 四种方阵配置，验证 CMM 等效模型对阵列规模的敏感性。

## 运行命令

### 1. 构建 CMM Array Size 条件模型

```bash
python -m src.test_cmm_array_size_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_array_size_conditions \
  --device cuda \
  --scope decoder_only \
  --rmin 1000 \
  --rmax 100000 \
  --array-sizes 64,128,256,512
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_array_size_conditions/conditions_manifest.json \
  --output checkpoints/cmm_array_size_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_array_size_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_array_size_conditions | 用户记录 | checkpoints/cmm_array_size_conditions | array size 条件模型输出目录 |
| --device | cuda | 推断自同组 CMM 实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 推断自同组 CMM 实验环境 | decoder_only | CMM 映射范围 |
| --rmin | 1000 | 推断自同组 CMM 实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组 CMM 实验环境 | 100000.0 | Roff，器件最大电阻 |
| --array-sizes | 64,128,256,512 | 用户记录/默认值 | 64,128,256,512 | 扫描的方阵 tile 尺寸列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 该脚本与 CrossSim 版本结构一致，默认也固定 `cell_bits=0`、`write_noise_std=0.0`、`read_noise_std=0.0` 用于隔离纯 tile 切分效应。

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_array_size_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_array_size_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | tile_shape | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|------------|--------|--------|--------|---------|-------|-------|
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| tile-64×64 | 64×64 | 0.6831 | 0.2479 | 0.2282 | 0.4959 | 0.8524 | 0.1676 |
| tile-128×128 | 128×128 | 0.6818 | 0.2491 | 0.2293 | 0.4978 | 0.8517 | 0.1671 |
| tile-256×256 | 256×256 | 0.6805 | 0.2470 | 0.2284 | 0.4967 | 0.8543 | 0.1672 |
| tile-512×512 | 512×512 | 0.6812 | 0.2484 | 0.2293 | 0.4976 | 0.8544 | 0.1673 |

## 备注

- 不同 tile 尺寸下 caption 指标几乎不变：四组条件的 BLEU-4 均在 0.2470~0.2491 之间，与 baseline 的差异都非常小，属于评估采样波动范围。
- 不存在单调退化趋势：指标有轻微波动但没有随着阵列增大或减小而持续恶化，说明当前 CMM 等效模型对 tile 切分不敏感。
- 128×128 与 512×512 略优但不具统计意义，更像随机波动而非真实硬件收益。
- 与 CrossSim array size 消融结果完全一致：在无噪声、连续 cell state 的理想条件下，阵列切分本身不会成为 caption 质量瓶颈。
- 注意：当前 PyTorch CMM 中 tile 仅影响实现切分方式，不会引入真实 crossbar 中的 IR drop、失配累积等物理非理想效应。后续需在 CrossSim 接入阶段验证大阵列是否会带来这些额外因素。
- 结论：当前 CMM 中 array size 不是精度瓶颈，默认 128×128 可以继续沿用。
- 仍缺失的信息：精确执行时间。
