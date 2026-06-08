# CMM-CrossSim DAC 分辨率消融实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-26（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/test_cmm_crosssim_dac_conditions.py`；`src/compute_metrics_pycoco.py` |

## 实验目的

在 CMM → CrossSim 两级映射路径上系统评估 DAC 量化精度对 caption 质量的影响。该实验同时包含一个关键对照：在同样 DAC 扫描下将 write/read noise 设为 0，隔离验证当前退化到底是由 `1e-4` 读写噪声耦合造成，还是由 DAC 量化本身直接导致。

## 运行命令

### 1. 构建 CMM-CrossSim DAC 条件模型（含 1e-4 读写噪声）

```bash
python -m src.test_cmm_crosssim_dac_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_crosssim_dac_conditions \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --adc-resolution 0 \
  --read-noise-std 1e-4 \
  --write-noise-std 1e-4 \
  --dac-resolutions 12,10,8,6,4
```

### 2. 构建 CMM-CrossSim DAC 条件模型（无噪声对照）

```bash
python -m src.test_cmm_crosssim_dac_conditions \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output-dir checkpoints/cmm_crosssim_dac_conditions_no_noise \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --adc-resolution 0 \
  --read-noise-std 0.0 \
  --write-noise-std 0.0 \
  --dac-resolutions 12
```

### 3. 批量计算 pycocoevalcap 指标

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_crosssim_dac_conditions/conditions_manifest.json \
  --output checkpoints/cmm_crosssim_dac_conditions/metrics_pycoco.json \
  --device cuda \
  --max-len 30 \
  --limit 500
```

## 参数说明

### 脚本 1：`src/test_cmm_crosssim_dac_conditions.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_crosssim_dac_conditions | 用户记录 | checkpoints/cmm_crosssim_dac_conditions | DAC 条件模型输出目录 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 映射与推理设备 |
| --use-gpu | False | 默认值 | False | CrossSim GPU 后端 |
| --scope | decoder_only | 用户记录/默认值 | decoder_only | CMM-on-CrossSim 映射范围 |
| --tile-rows | 128 | 推断自同组实验环境 | 128 | 阵列最大行数 |
| --tile-cols | 128 | 推断自同组实验环境 | 128 | 阵列最大列数 |
| --rmin | 1000 | 推断自同组实验环境 | 1000.0 | Ron |
| --rmax | 100000 | 推断自同组实验环境 | 100000.0 | Roff |
| --cell-bits | 0 | 默认值 | 0 | 连续电导 |
| --adc-resolution | 0 | 默认值 | 0 | 固定 ADC 为理想，隔离纯 DAC 效应 |
| --read-noise-std | 1e-4（主实验）/ 0.0（对照） | 默认值 1e-4 / 用户指定 0 | 1e-4 | 读噪声 |
| --write-noise-std | 1e-4（主实验）/ 0.0（对照） | 默认值 1e-4 / 用户指定 0 | 1e-4 | 写入噪声 |
| --bias-rows | 0 | 默认值 | 0 | bias 映射额外行数 |
| --dac-resolutions | 12,10,8,6,4 | 用户记录/默认值 | 12,10,8,6,4 | 扫描的 DAC bit 列表 |
| --skip-existing | False | 默认值 | False | 若目标 checkpoint 已存在则跳过 |

> 脚本默认在 `write_noise_std=1e-4`、`read_noise_std=1e-4` 的非零噪声基线下运行。[src/test_cmm_crosssim_dac_conditions.py:1-6](src/test_cmm_crosssim_dac_conditions.py#L1-L6)

### 脚本 2：`src/compute_metrics_pycoco.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_crosssim_dac_conditions/conditions_manifest.json | 用户记录 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_crosssim_dac_conditions/metrics_pycoco.json | 用户记录 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --device | cuda | 推断自同组实验环境 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --limit | 500 | 用户记录 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |

## 实验结果

### pycocoevalcap 全指标评估（含 1e-4 读写噪声, `compute_metrics_pycoco.py`, limit=500）

| 条件 | ADC bits | DAC bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|----------|----------|--------|--------|--------|---------|-------|-------|
| baseline | — | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| dac-12bit | 0 | 12 | 0.5812 | 0.1614 | 0.1756 | 0.4337 | 0.5380 | 0.1235 |
| dac-10bit | 0 | 10 | 0.5811 | 0.1627 | 0.1764 | 0.4345 | 0.5405 | 0.1240 |
| dac-8bit | 0 | 8 | 0.5786 | 0.1576 | 0.1756 | 0.4340 | 0.5413 | 0.1226 |
| dac-6bit | 0 | 6 | 0.5409 | 0.1471 | 0.1674 | 0.4276 | 0.5216 | 0.1205 |
| dac-4bit | 0 | 4 | 0.2066 | 0.0420 | 0.0654 | 0.2599 | 0.1713 | 0.0589 |

### 无噪声对照（`checkpoints/cmm_crosssim_dac_conditions_no_noise/metrics_pycoco.json`, dac-12 单点）

| 条件 | write/read noise | BLEU-4 | CIDEr |
|------|-----------------|--------|-------|
| dac-12bit + noise 1e-4 | 1e-4 / 1e-4 | 0.1614 | 0.5380 |
| dac-12bit + no noise | 0.0 / 0.0 | 0.1637 | 0.5397 |

## 备注

- 本实验默认在 `write_noise_std=1e-4`、`read_noise_std=1e-4` 条件下运行，DAC 效应叠加在轻度读写噪声之上评估。[src/test_cmm_crosssim_dac_conditions.py:1-6](src/test_cmm_crosssim_dac_conditions.py#L1-L6)
- 即使 `dac-12bit` / `dac-10bit` 也已经显著退化：BLEU-4 仅约 0.16（baseline 0.2464），CIDEr 仅约 0.54（baseline 0.8546），说明在 CMM-CrossSim 路径下 DAC 的敏感性远高于纯 PyTorch CMM。
- `dac-12 / dac-10 / dac-8` 三组差异极小：BLEU-4 都在 0.158~0.163、CIDEr 都在 0.538~0.541，说明退化进入了 DAC 量化误差主导的平台区，仅靠提高位宽无法回复。
- `dac-6bit` 进一步退化，`dac-4bit` 接近功能失效（BLEU-4 仅 0.0420）。
- 无噪声对照关键发现：在 `write_noise_std=0`、`read_noise_std=0` 下，`dac-12` 的 BLEU-4 仍只有 0.1637、CIDEr 仅 0.5397，与含 `1e-4` 噪声条件几乎一致。这表明当前退化主要由 DAC 量化本身引起，而非 `1e-4` 级别读写噪声耦合造成。
- 跨模型对比：
  - 纯 PyTorch CMM：dac-10/8/6 基本无损
  - CrossSim 原生：dac-12 已有明显退化
  - CMM-CrossSim：dac-12 也有显著退化，结果更接近 CrossSim 原生模型
- 这进一步支持核心判断：PyTorch CMM 等效模型过于温和；一旦放入 CrossSim 真实器件仿真，DAC 非理想性会迅速成为主导瓶颈，其敏感性甚至强于同一路径下的 ADC 消融结果。
- 结论：CMM-CrossSim 路径下 DAC 是极强敏感项，即使 12-bit 也无法保持 baseline 水平；推荐 DAC ≥ 12-bit 并探索非量化层面（如器件模型、映射策略、误差补偿）的缓解手段。
- 仍缺失的信息：精确执行时间；无噪声对照实验是否仅跑了 dac-12 单点，还是对全量程都做了对照。
