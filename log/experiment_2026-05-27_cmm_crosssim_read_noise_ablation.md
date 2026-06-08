# CMM-CrossSim Read Noise 消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-27（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_crosssim_read_noise_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

在 CMM → CrossSim 两级映射路径上系统评估 CrossSim read noise 强度对 caption 质量的影响，通过 `seed=1,2,3` 三种子捕捉噪声的种子间方差。验证 read noise 在该路径下是否仍像纯 CMM 和纯 CrossSim 中表现的那样基本无害。

## 运行命令

### 1. 构建 CMM-CrossSim Read Noise 条件模型

```bash
python -m src.test_cmm_crosssim_read_noise_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_crosssim_read_noise_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --adc-resolution 0 \
  --dac-resolution 0 \
  --write-noise-std 1e-4 \
  --read-noise-stds 0,1e-4,3e-4,1e-3,3e-3,1e-2 \
  --seeds 1,2,3
```

### 2. 批量计算 pycocoevalcap 指标（含 seed 分组求 mean±std）

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_crosssim_read_noise_conditions/conditions_manifest.json \
  --output checkpoints/cmm_crosssim_read_noise_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_crosssim_read_noise_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_crosssim_read_noise_conditions | 默认值 | checkpoints/cmm_crosssim_read_noise_conditions | read noise 条件模型输出目录 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | False | 默认值 | False | CrossSim GPU 后端 |
| --scope | decoder_only | 默认值 | decoder_only | CMM-on-CrossSim 映射范围 |
| --tile-rows | 128 | 默认值 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 默认值 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 默认值 | 1000.0 | Ron |
| --rmax | 100000 | 默认值 | 100000.0 | Roff |
| --cell-bits | 0 | 默认值 | 0 | 连续电导 |
| --adc-resolution | 0 | 默认值 | 0 | 固定 ADC 为理想，隔离纯 read noise 效应 |
| --dac-resolution | 0 | 默认值 | 0 | 固定 DAC 为理想 |
| --write-noise-std | 1e-4 | 默认值 | 1e-4 | 固定写入噪声 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射额外行数 |
| --read-noise-stds | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 默认值 | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 扫描的读噪声标准差列表 |
| --seeds | 1,2,3 | 默认值 | 1,2,3 | 每个噪声强度使用的随机种子列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本内部硬固定：`tile_rows=128`、`tile_cols=128`、`cell_bits=0`、`adc_resolution=0`、`dac_resolution=0`、`write_noise_std=1e-4`，确保实验变量仅为 `read_noise_std` 和 `seed`。

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_crosssim_read_noise_conditions/conditions_manifest.json | 用户指定 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_crosssim_read_noise_conditions/metrics_pycoco.json | 用户指定 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户指定 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（`compute_metrics_pycoco.py`, limit=500, n=3 seeds）

| read_noise_std | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|----------------|--------|--------|--------|---------|-------|-------|
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| 0 | 0.6820 ± 0.0013 | 0.2471 ± 0.0008 | 0.2277 ± 0.0003 | 0.4953 ± 0.0006 | 0.8552 ± 0.0023 | 0.1682 ± 0.0005 |
| 1e-4 | 0.6830 ± 0.0012 | 0.2492 ± 0.0013 | 0.2283 ± 0.0005 | 0.4962 ± 0.0007 | 0.8575 ± 0.0025 | 0.1682 ± 0.0004 |
| 3e-4 | 0.6827 ± 0.0010 | 0.2483 ± 0.0015 | 0.2280 ± 0.0005 | 0.4958 ± 0.0009 | 0.8562 ± 0.0030 | 0.1679 ± 0.0004 |
| 1e-3 | 0.6827 ± 0.0007 | 0.2498 ± 0.0005 | 0.2283 ± 0.0001 | 0.4961 ± 0.0001 | 0.8579 ± 0.0011 | 0.1683 ± 0.0001 |
| 3e-3 | 0.6834 ± 0.0006 | 0.2489 ± 0.0007 | 0.2285 ± 0.0004 | 0.4964 ± 0.0002 | 0.8572 ± 0.0004 | 0.1684 ± 0.0003 |
| 1e-2 | 0.6836 ± 0.0009 | 0.2509 ± 0.0028 | 0.2286 ± 0.0004 | 0.4965 ± 0.0008 | 0.8557 ± 0.0029 | 0.1680 ± 0.0007 |

### 按 noise 组 × seed 的详细指标

| 条件 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|--------|--------|--------|---------|-------|-------|
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| read-noise-0_seed-1 | 0.6831 | 0.2483 | 0.2281 | 0.4960 | 0.8581 | 0.1688 |
| read-noise-0_seed-2 | 0.6802 | 0.2465 | 0.2273 | 0.4946 | 0.8524 | 0.1677 |
| read-noise-0_seed-3 | 0.6826 | 0.2467 | 0.2278 | 0.4954 | 0.8549 | 0.1680 |
| read-noise-1e-04_seed-1 | 0.6847 | 0.2509 | 0.2289 | 0.4971 | 0.8609 | 0.1687 |
| read-noise-1e-04_seed-2 | 0.6818 | 0.2478 | 0.2278 | 0.4954 | 0.8548 | 0.1682 |
| read-noise-1e-04_seed-3 | 0.6824 | 0.2490 | 0.2281 | 0.4962 | 0.8567 | 0.1678 |
| read-noise-3e-04_seed-1 | 0.6841 | 0.2504 | 0.2287 | 0.4969 | 0.8605 | 0.1684 |
| read-noise-3e-04_seed-2 | 0.6822 | 0.2469 | 0.2276 | 0.4948 | 0.8537 | 0.1675 |
| read-noise-3e-04_seed-3 | 0.6819 | 0.2477 | 0.2276 | 0.4957 | 0.8543 | 0.1678 |
| read-noise-1e-03_seed-1 | 0.6826 | 0.2494 | 0.2283 | 0.4960 | 0.8577 | 0.1684 |
| read-noise-1e-03_seed-2 | 0.6836 | 0.2504 | 0.2284 | 0.4961 | 0.8594 | 0.1684 |
| read-noise-1e-03_seed-3 | 0.6820 | 0.2495 | 0.2282 | 0.4961 | 0.8568 | 0.1682 |
| read-noise-3e-03_seed-1 | 0.6825 | 0.2485 | 0.2287 | 0.4967 | 0.8578 | 0.1683 |
| read-noise-3e-03_seed-2 | 0.6839 | 0.2499 | 0.2289 | 0.4965 | 0.8567 | 0.1687 |
| read-noise-3e-03_seed-3 | 0.6839 | 0.2482 | 0.2279 | 0.4961 | 0.8572 | 0.1681 |
| read-noise-1e-02_seed-1 | 0.6831 | 0.2528 | 0.2288 | 0.4970 | 0.8598 | 0.1689 |
| read-noise-1e-02_seed-2 | 0.6828 | 0.2469 | 0.2281 | 0.4953 | 0.8542 | 0.1677 |
| read-noise-1e-02_seed-3 | 0.6849 | 0.2530 | 0.2290 | 0.4972 | 0.8531 | 0.1673 |

## 备注

- 本实验默认在 `write_noise_std=1e-4`、ADC/DAC=0/0、cell_bits=0 的轻度非理想基线下运行，read noise 作为唯一扫描变量叠加评估。
- CMM-CrossSim read noise 在所有级别下几乎无影响：即使 `std=1e-2`，BLEU-4 均值为 0.2509（甚至比 baseline 0.2464 高 +1.8%），CIDEr 均值 0.8557 与 baseline 0.8546 基本持平。
- `1e-3 ~ 1e-2` 区间有微弱正效应：BLEU-4 均值在 0.2498~0.2509，略优于 baseline。物理直觉：瞬时 read noise 不改变存储状态，仅影响个别读取值，在序列生成中会被多 token 平均自抵消。
- 种子间方差极小：BLEU-4 最大 std 仅 0.0028（`1e-2`），CIDEr 最大 std 仅 0.0030（`3e-4`），远小于同级别的 CMM-CrossSim write noise 方差。
- SPICE 极度稳定：全量程 mean 波动 < 0.0005，std ≤ 0.0007。
- 与纯 CMM read noise 和 CrossSim 原生 read noise 结果高度一致：三种模型路径共同确认 read noise 对 Transformer 图像描述任务基本无害，主要精度瓶颈仍来自写入端（write noise / cell_bits / DAC）。
- 零读噪声三组结果（read-noise-0）与 baseline 指标基本一致，确认 CMM-CrossSim 映射在仅含 1e-4 write noise 时未引入额外系统性偏差。
- 结论：CMM-CrossSim 路径下 read noise 基本无害，推荐关注写入端非理想因素的缓解。
- 仍缺失的信息：精确执行时间。
