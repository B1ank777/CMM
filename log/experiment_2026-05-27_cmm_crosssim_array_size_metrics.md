# CMM-on-CrossSim Array Size 消融实验 (ADC=10) — pycoco 指标评测

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-28 |
| **状态** | 已完成 |
| **脚本** | src/test_cmm_crosssim_array_size_conditions.py, src/compute_metrics_pycoco.py |

## 实验目的

在 CMM-on-CrossSim 路径上，固定 ADC=10（中等非理想量化）、DAC=0（理想）、cell_bits=0、write/read noise=1e-4 的条件下，扫描四种 tile 尺寸（64×64, 128×128, 256×256, 512×512），评测阵列尺寸与 ADC 量化的交叉效应对描述生成质量的影响。

上一轮实验中 ADC=0 时所有 array size 均与基线几乎一致，本轮将 ADC 提高至 10 bit，观察 tile 划分在 ADC 量化存在时是否表现出瓶颈效应。

## 运行命令

```bash
# 步骤 1：生成条件模型（ADC=10）
python -m src.test_cmm_crosssim_array_size_conditions \
    --checkpoint checkpoints\caption_transformer_epoch_10.pt \
    --output-dir checkpoints\cmm_crosssim_array_size_conditions \
    --adc-resolution 10

# 步骤 2：评测指标
python -m src.compute_metrics_pycoco \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --conditions-manifest checkpoints/cmm_crosssim_array_size_conditions/conditions_manifest.json \
    --output checkpoints/cmm_crosssim_array_size_conditions/metrics_pycoco.json \
    --limit 500
```

## 参数说明

### test_cmm_crosssim_array_size_conditions.py

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | (required) | 原始训练检查点 |
| --output-dir | checkpoints/cmm_crosssim_array_size_conditions | 用户指定 | checkpoints/cmm_crosssim_array_size_conditions | 输出目录 |
| --adc-resolution | 10 | 用户指定 | 0 | ADC 分辨率（覆盖了此前硬编码的 0） |
| --dac-resolution | 0 | 默认值 | 0 | DAC 分辨率 |
| --cell-bits | 0 | 默认值 | 0 | 单元量化 bit 数 |
| --read-noise-std | 1e-4 | 默认值 | 1e-4 | 读噪声强度 |
| --write-noise-std | 1e-4 | 默认值 | 1e-4 | 写入噪声强度 |
| --seed | 42 | 默认值 | 42 | 随机种子 |
| --array-sizes | 64,128,256,512 | 默认值 | 64,128,256,512 | 方阵尺寸列表 |

### compute_metrics_pycoco.py

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | (required) | 基线模型检查点 |
| --conditions-manifest | checkpoints/cmm_crosssim_array_size_conditions/conditions_manifest.json | 用户指定 | (required) | 条件清单 |
| --output | checkpoints/cmm_crosssim_array_size_conditions/metrics_pycoco.json | 用户指定 | checkpoints/metrics_pycoco.json | 输出路径 |
| --limit | 500 | 用户指定 | 1000 | 评测图片上限 |
| --coco-root | data/coco | 默认值 | data/coco | COCO 数据集根目录 |
| --device | cuda | 默认值 | cuda (if available) | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |

## 条件清单

| condition | tile_rows | tile_cols | adc_resolution | dac_resolution | cell_bits | write_noise | read_noise | seed |
|-----------|-----------|-----------|----------------|----------------|-----------|-------------|------------|------|
| array-64x64 | 64 | 64 | **10** | 0 | 0 | 0.0001 | 0.0001 | 42 |
| array-128x128 | 128 | 128 | **10** | 0 | 0 | 0.0001 | 0.0001 | 42 |
| array-256x256 | 256 | 256 | **10** | 0 | 0 | 0.0001 | 0.0001 | 42 |
| array-512x512 | 512 | 512 | **10** | 0 | 0 | 0.0001 | 0.0001 | 42 |

## 实验结果

| model | BLEU-1 | BLEU-4 | METEOR | ROUGE_L | CIDEr | SPICE |
|-------|--------|--------|--------|---------|-------|-------|
| baseline (digital) | 0.6821 | 0.2481 | 0.2279 | 0.4957 | 0.8560 | 0.1682 |
| array-64x64 | 0.6772 | 0.2506 | 0.2298 | 0.4945 | 0.8575 | 0.1707 |
| array-128x128 | 0.6665 | 0.2389 | 0.2245 | 0.4889 | 0.8396 | 0.1604 |
| array-256x256 | 0.6737 | 0.2361 | 0.2267 | 0.4922 | 0.8184 | 0.1640 |
| array-512x512 | 0.6824 | 0.2414 | 0.2274 | 0.4944 | 0.8314 | 0.1658 |

### 与上一轮 (ADC=0) 的对比

| model | BLEU-4 (ADC=0) | BLEU-4 (ADC=10) | Δ | CIDEr (ADC=0) | CIDEr (ADC=10) | Δ |
| ----- | --------------- | ---------------- | --- | ------------- | --------------- | --- |
| array-64x64 | 0.2494 | 0.2506 | +0.0012 | 0.8595 | 0.8575 | -0.0020 |
| array-128x128 | 0.2481 | 0.2389 | **-0.0092** | 0.8550 | 0.8396 | **-0.0154** |
| array-256x256 | 0.2496 | 0.2361 | **-0.0135** | 0.8581 | 0.8184 | **-0.0397** |
| array-512x512 | 0.2471 | 0.2414 | -0.0057 | 0.8556 | 0.8314 | -0.0242 |

### 关键发现

1. **ADC=10 使 array size 效应显现**：与 ADC=0 时所有 tile 尺寸几乎无损不同，ADC=10 下出现了明显的性能分化。array-128×128 和 array-256×256 退化最显著（BLEU-4 下降约 0.01~0.014，CIDEr 下降约 0.015~0.04），array-512×512 次之，array-64×64 几乎不受影响。

2. **非单调退化模式**：array-64×64 在 ADC=10 下依然紧贴基线（BLEU-4=0.2506，甚至略高于 basline 的 0.2481），而中间尺寸 128×128 和 256×256 退化最大，512×512 有所恢复。这可能说明两种对抗效应的叠加：
   - 大 tile → 每次 ADC 转换前累积的模拟部分和更大 → 量化误差相对信号更小 → 有利
   - 小 tile → 更多 ADC 转换次数 → 单次量化误差经多次平均后相互抵消 → 也可能有利
   - 中间尺寸落入"两头不沾"的尴尬区

3. **64×64 的鲁棒性值得进一步验证**：在 ADC=0 和 ADC=10 两个条件下 array-64×64 始终与基线持平，暗示细粒度 tiling + ADC 量化的噪声平均效应可能存在理论最优解。但 BLEU-4 高于基线 0.0025 的量级在自回归生成方差范围内，不宜过度解读为"超越基线"。

4. **CIDEr 比 BLEU-4 对 ADC 量化更敏感**：array-256×256 的 CIDEr 下降 0.0397（约 4.6%）而 BLEU-4 仅下降 0.0135，说明 ADC 量化主要伤害的是语义一致性而非 n-gram 匹配。

## 备注

- 评测图片 500 张，image_id 范围 [391895, 289423]。
- 每个条件映射 21 个 nn.Linear 层。
- 本次实验前修复了脚本 bug：原 `build_condition_models()` 硬编码 `args.adc_resolution = 0` 覆盖命令行参数，manifest 也硬编码字符串 `"0"`。修复后命令行 `--adc-resolution 10` 正常生效。
- 后续可考虑：在更高 ADC 非理想水平（如 adc-8、adc-6）下重复 array size 扫描，验证非单调模式的稳定性；或增加重复 seed 跑 3 次评估标准差。
