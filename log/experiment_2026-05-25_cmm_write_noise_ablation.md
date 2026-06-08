# CMM Write Noise 消融实验（多种子）

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_write_noise_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估 CMM 映射中写入噪声强度对 caption 质量的影响，并通过 `seed=1,2,3` 三种子捕捉噪声注入的种子间方差，确认 CMM 的 `r`-state clamp `[0,1]` 机制是否确实天然抑制了写入噪声的逐层累积。

## 运行命令

### 1. 构建 CMM Write Noise 条件模型

```bash
python -m src.test_cmm_write_noise_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_write_noise_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --write-noise-stds 0,1e-4,3e-4,1e-3,3e-3,1e-2 \
  --seeds 1,2,3
```

### 2. 批量计算 pycocoevalcap 指标（含 seed 分组求 mean±std）

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_write_noise_conditions/conditions_manifest.json \
  --output checkpoints/cmm_write_noise_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_write_noise_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_write_noise_conditions | 用户记录 | checkpoints/cmm_write_noise_conditions | write noise 条件模型输出目录 |
| --device | cuda | 推断自同组 CMM 实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 推断自同组 CMM 实验环境 | decoder_only | CMM 映射范围 |
| --tile-rows | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 推断自同组 CMM 实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组 CMM 实验环境 | 100000.0 | Roff，器件最大电阻 |
| --write-noise-stds | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 用户记录/默认值 | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 扫描的写入噪声标准差列表 |
| --seeds | 1,2,3 | 用户记录/默认值 | 1,2,3 | 每个噪声强度使用的随机种子列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本内部固定：cell_bits=0、read_noise_std=0.0，用于隔离纯 write noise 效应。[src/test_cmm_write_noise_conditions.py:112-114](src/test_cmm_write_noise_conditions.py#L112-L114)

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_write_noise_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_write_noise_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500, n=3 seeds）

| write_noise_std | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|-----------------|--------|--------|--------|---------|-------|-------|
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| 0 | 0.6816 ± 0 | 0.2464 ± 0 | 0.2276 ± 0 | 0.4954 ± 0 | 0.8546 ± 0 | 0.1680 ± 0 |
| 1e-4 | 0.6823 ± 0.0004 | 0.2477 ± 0.0017 | 0.2278 ± 0.0005 | 0.4957 ± 0.0007 | 0.8548 ± 0.0035 | 0.1678 ± 0.0001 |
| 3e-4 | 0.6825 ± 0.0013 | 0.2487 ± 0.0012 | 0.2280 ± 0.0005 | 0.4961 ± 0.0004 | 0.8539 ± 0.0021 | 0.1673 ± 0.0002 |
| 1e-3 | 0.6816 ± 0.0050 | 0.2481 ± 0.0019 | 0.2278 ± 0.0009 | 0.4962 ± 0.0013 | 0.8540 ± 0.0004 | 0.1670 ± 0.0006 |
| 3e-3 | 0.6787 ± 0.0056 | 0.2444 ± 0.0035 | 0.2267 ± 0.0007 | 0.4956 ± 0.0006 | 0.8490 ± 0.0008 | 0.1671 ± 0.0008 |
| 1e-2 | 0.6739 ± 0.0133 | 0.2410 ± 0.0054 | 0.2245 ± 0.0019 | 0.4914 ± 0.0029 | 0.8395 ± 0.0070 | 0.1648 ± 0.0010 |

## 备注

- 写入噪声在 CMM 映射时固定注入到内部状态 `r_pos/r_neg`，随 checkpoint 固化，推理时不再变动。[src/test_cmm_write_noise_conditions.py:1-7](src/test_cmm_write_noise_conditions.py#L1-L7)
- 每个噪声强度跑 3 个 seed，用于评估噪声的种子间方差。指标报告为 `mean ± std`。
- CMM 写入噪声影响远小于 CrossSim：即使 `std=1e-2`，BLEU-4 仅从 0.2464 降至 0.2410（-2.2%），而 CrossSim write noise 在 `it-6 (1e-2)` 时 BLEU-4=0、完全崩溃。CMM 的 `r`-state clamp `[0,1]` 机制天然抑制了噪声的逐层累积。
- `1e-4 ~ 1e-3` 区间有微弱正效应：BLEU-4 均值略高于 baseline（0.2477~0.2487 vs 0.2464），与 CrossSim 中 read noise 的正则化效应类似。
- 噪声 ≥ 1e-3 时种子间方差开始明显：BLEU-1 std 从 0.0004 逐步增至 0.0133。
- SPICE 在各噪声级别均保持极低波动（std ≤ 0.0010），对 write noise 最不敏感。
- 结论：CMM write noise 在现有参数下对模型质量影响极弱，即使 1e-2 也远未达到 CrossSim 同等破坏力。后续需在 CrossSim 接入阶段以真实器件模型验证该发现是否成立。
- 仍缺失的信息：精确执行时间。
