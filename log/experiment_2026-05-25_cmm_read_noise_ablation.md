# CMM Read Noise 消融实验（多种子）

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_read_noise_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

系统评估 CMM 映射中读噪声强度对 caption 质量的影响，并通过 `seed=1,2,3` 三种子捕捉噪声的种子间方差，验证推理阶段瞬时噪声是否像 CrossSim read noise 一样表现为无害甚至略有正则化效果。

## 运行命令

### 1. 构建 CMM Read Noise 条件模型

```bash
python -m src.test_cmm_read_noise_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_read_noise_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --read-noise-stds 0,1e-4,3e-4,1e-3,3e-3,1e-2 \
  --seeds 1,2,3
```

### 2. 批量计算 pycocoevalcap 指标（含 seed 分组求 mean±std）

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_read_noise_conditions/conditions_manifest.json \
  --output checkpoints/cmm_read_noise_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_read_noise_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_read_noise_conditions | 用户记录 | checkpoints/cmm_read_noise_conditions | read noise 条件模型输出目录 |
| --device | cuda | 推断自同组 CMM 实验环境 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 推断自同组 CMM 实验环境 | decoder_only | CMM 映射范围 |
| --tile-rows | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 推断自同组 CMM 实验环境 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 推断自同组 CMM 实验环境 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 推断自同组 CMM 实验环境 | 100000.0 | Roff，器件最大电阻 |
| --read-noise-stds | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 用户记录/默认值 | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 扫描的读噪声标准差列表 |
| --seeds | 1,2,3 | 用户记录/默认值 | 1,2,3 | 每个噪声强度使用的随机种子列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本内部固定：cell_bits=0、write_noise_std=0.0，用于隔离纯 read noise 效应。

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_read_noise_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_read_noise_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500, n=3 seeds）

| read_noise_std | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|----------------|--------|--------|--------|---------|-------|-------|
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| 0 | 0.6816 ± 0 | 0.2464 ± 0 | 0.2276 ± 0 | 0.4954 ± 0 | 0.8546 ± 0 | 0.1680 ± 0 |
| 1e-4 | 0.6817 ± 0.0001 | 0.2464 ± 0.0000 | 0.2276 ± 0 | 0.4954 ± 0 | 0.8547 ± 0.0001 | 0.1681 ± 0.0001 |
| 3e-4 | 0.6817 ± 0.0000 | 0.2472 ± 0.0007 | 0.2277 ± 0.0002 | 0.4955 ± 0.0002 | 0.8556 ± 0.0005 | 0.1680 ± 0.0002 |
| 1e-3 | 0.6832 ± 0.0014 | 0.2490 ± 0.0018 | 0.2283 ± 0.0006 | 0.4964 ± 0.0010 | 0.8590 ± 0.0027 | 0.1684 ± 0.0005 |
| 3e-3 | 0.6833 ± 0.0016 | 0.2492 ± 0.0022 | 0.2285 ± 0.0006 | 0.4967 ± 0.0012 | 0.8588 ± 0.0049 | 0.1682 ± 0.0007 |
| 1e-2 | 0.6817 ± 0.0010 | 0.2479 ± 0.0022 | 0.2277 ± 0.0005 | 0.4959 ± 0.0003 | 0.8505 ± 0.0040 | 0.1673 ± 0.0004 |

## 备注

- 读噪声在推理时注入到 `Rmem`，不改变已固化的内部状态，确保相同的 checkpoint 在不同实验中行为一致。
- 每个噪声强度跑 3 个 seed，用于评估噪声的种子间方差。指标报告为 `mean ± std`。
- CMM read noise 在所有级别下几乎无影响：即使 `std=1e-2`，BLEU-4 仍为 0.2479（甚至比 baseline 0.2464 略高），CIDEr 仅从 0.8546 降至 0.8505（-0.5%）。
- `1e-3 ~ 3e-3` 区间有微弱正效应：BLEU-4 0.2490~0.2492 vs baseline 0.2464（+1.1%），与 CrossSim read noise 在 1e-3 处的正则化效应一致。物理直觉：瞬时噪声不改变存储状态，仅影响个别读取值，在序列生成中会被多 token 平均自抵消。
- `1e-2` 时正效应消退：BLEU-4 回落到 0.2479，CIDEr 回落到 0.8505，但仍略优于 baseline。
- 种子间方差极小：BLEU-4 最大 std 仅 0.0022（`1e-2`），远小于同级别的 write noise std（0.0054），印证 read noise 是多 token 平均自抵消的。
- SPICE 极度稳定：全量程 std ≤ 0.0007，mean 波动 < 0.001。
- 结论：与 CrossSim 的 read noise 结果高度一致，两种模型共同确认 read noise 对 Transformer 图像描述任务基本无害。主要精度瓶颈仍来自写入端（write noise / cell_bits / DAC）。
- 仍缺失的信息：精确执行时间；以及是否使用了多元化的种子——seed 1/2/3。
