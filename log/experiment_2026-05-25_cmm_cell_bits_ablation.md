# CMM cell_bits 量化消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_cell_bits_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估论文式 CMM 映射中 `cell_bits` 量化对 caption 质量的影响，判断连续忆阻器状态到低比特离散状态的压缩会在什么阈值开始显著损害模型性能，并据此给出可接受的 CMM 单元状态量化位宽建议。

## 运行命令

### 1. 构建 CMM cell_bits 条件模型

```bash
python -m src.test_cmm_cell_bits_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_cell_bits_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --write-noise-std 0.0 \
  --read-noise-std 0.0 \
  --cell-bits-list 0,2,3,4,6,8
```

### 2. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_cell_bits_conditions/conditions_manifest.json \
  --output checkpoints/cmm_cell_bits_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_cell_bits_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_cell_bits_conditions | 用户记录 | checkpoints/cmm_cell_bits_conditions | CMM cell_bits 条件模型输出目录 |
| --device | cuda | 推断自同组 CMM 实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 推断自同组 CMM 实验环境 | decoder_only | CMM 映射范围 |
| --tile-rows | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 推断自同组 CMM 实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组 CMM 实验环境 | 100000.0 | Roff，器件最大电阻 |
| --write-noise-std | 0.0 | 推断自同组 CMM 实验环境 | 0.0 | 写入噪声强度 |
| --read-noise-std | 0.0 | 推断自同组 CMM 实验环境 | 0.0 | 读噪声强度 |
| --cell-bits-list | 0,2,3,4,6,8 | 用户记录/默认值 | 0,2,3,4,6,8 | 扫描的 CMM cell_bits 列表；0 表示连续状态 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_cell_bits_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_cell_bits_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500）

| 条件 | cell_bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|-----------|--------|--------|--------|---------|-------|-------|
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| cell-continuous | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| cell-8bit | 8 | 0.6783 | 0.2455 | 0.2280 | 0.4951 | 0.8503 | 0.1668 |
| cell-6bit | 6 | 0.6721 | 0.2443 | 0.2254 | 0.4902 | 0.8377 | 0.1643 |
| cell-4bit | 4 | 0.6528 | 0.2266 | 0.2243 | 0.4892 | 0.8018 | 0.1623 |
| cell-3bit | 3 | 0.5621 | 0.1708 | 0.1974 | 0.4239 | 0.6624 | 0.1418 |
| cell-2bit | 2 | 0.0120 | ~0 | 0.0140 | 0.0175 | ~0 | 0.0145 |

## 备注

- 该脚本不依赖 CrossSim simulator，而是直接使用论文式 CMM 映射；默认扫描 `cell_bits = 0, 2, 3, 4, 6, 8`，其中 `0` 表示连续忆阻器状态。[src/test_cmm_cell_bits_conditions.py:1-6](src/test_cmm_cell_bits_conditions.py#L1-L6)
- 条件名由脚本自动生成：`0` 对应 `cell-continuous`，其余位宽对应 `cell-{n}bit`。[src/test_cmm_cell_bits_conditions.py:88-91](src/test_cmm_cell_bits_conditions.py#L88-L91)
- `cell-continuous` 与 baseline 完全一致，说明连续状态下 CMM 映射保持无损，也与前面的等价性/decoder_only 验证结论一致。
- `cell-8bit` 几乎无损：BLEU-4 仅从 `0.2464` 降到 `0.2455`，CIDEr 仅从 `0.8546` 降到 `0.8503`。
- `cell-6bit` 为轻微退化区：CIDEr 下降约 2%，BLEU-4 下降不足 1%，整体仍可接受。
- `cell-4bit` 开始出现明显退化：BLEU-4 降到 `0.2266`，CIDEr 降到 `0.8018`。
- `cell-3bit` 已严重退化，`cell-2bit` 则基本功能崩溃，说明低比特状态已无法有效承载原始权重信息。
- 结论：推荐 `cell_bits ≥ 6`；其中 `8-bit` 基本保持全精度，`6-bit` 仍可接受，`4-bit` 是退化起点。
- 与 2026-05-22 的 CrossSim ADC 消融相比，CMM `cell_bits` 的退化曲线更平滑，崩溃阈值更低级别地出现在 `2-bit` 而非中高位宽。
- 仍缺失的信息：这组 sweep 的精确执行时间；以及是否还计算过 `metrics_mean_std` 一类多次重复评估结果，还是当前只保存了单次 `metrics_pycoco.json`。
