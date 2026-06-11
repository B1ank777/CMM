# DAC 退化根因分析 — Clipping Rate + Adaptive DAC 对照实验

| 属性 | 值 |
|------|-----|
| **日期** | 2026-06-10 21:27 |
| **状态** | 已完成 |
| **脚本** | analyze_cmm_crosssim_dac_clipping.py, evaluate_cmm_crosssim_adaptive_dac.py |
| **辅助模块** | dac_adaptive_utils.py |

## 实验目的

验证审稿人提出的假设：CMM-on-CrossSim DAC 退化主因不是量化分辨率，而是 CrossSim `QuantizerDAC` 固定 [-1, 1] 输入范围导致大量 activation 被 clipping。分三步验证：

1. **实验 1**：统计 21 层 mapped Linear 输入 activation 的分布和 clip_rate（`|x| > 1` 比例）
2. **实验 2**：画典型层原始/固定裁剪/自适应量化三条分布对比图
3. **实验 3**：评测三种 DAC-12 设置（current [-1,1]、layer-wise adaptive、batch-wise adaptive）的 BLEU/CIDEr

## 运行命令

```bash
# 实验 1+2: activation clipping 统计 + 分布图
python -m src.analyze_cmm_crosssim_dac_clipping \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --crosssim-checkpoint checkpoints/cmm_crosssim_dac_conditions/caption_transformer_dac-12bit_cmm_crosssim.pt \
    --coco-root data/coco \
    --output-dir experiments/dac_range_root_cause \
    --device cuda --max-batches 50 --dac-bits 12

# 实验 3: adaptive DAC 三设置对照
python -m src.evaluate_cmm_crosssim_adaptive_dac \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --crosssim-checkpoint checkpoints/cmm_crosssim_dac_conditions/caption_transformer_dac-12bit_cmm_crosssim.pt \
    --coco-root data/coco \
    --output experiments/dac_range_root_cause/adaptive_dac12_metrics.json \
    --device cuda --limit 1000 --calibration-batches 50 --batch-size 32
```

## 参数说明

### analyze_cmm_crosssim_dac_clipping.py

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 默认值 | checkpoints/caption_transformer_epoch_10.pt | 数字基线 checkpoint |
| --crosssim-checkpoint | .../caption_transformer_dac-12bit_cmm_crosssim.pt | 默认值 | .../caption_transformer_dac-12bit_cmm_crosssim.pt | DAC-12 CMM-CrossSim checkpoint |
| --coco-root | data/coco | 默认值 | data/coco | COCO 数据根目录 |
| --output-dir | experiments/dac_range_root_cause | 默认值 | experiments/dac_range_root_cause | 输出目录 |
| --device | cuda | 用户指定 (默认值) | cuda (if available) | 推理设备 |
| --max-batches | 50 | 默认值 | 50 | 最多采样 batch 数 |
| --dac-bits | 12 | 默认值 | 12 | 画 adaptive 量化分布用的 DAC bit |
| --batch-size | 8 | 默认值 | 8 | activation 采样 batch size |
| --activation-source | baseline | 默认值 | baseline | 统计来源（量化前原始分布） |

### evaluate_cmm_crosssim_adaptive_dac.py

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 默认值 | checkpoints/caption_transformer_epoch_10.pt | 数字基线 checkpoint |
| --crosssim-checkpoint | .../caption_transformer_dac-12bit_cmm_crosssim.pt | 默认值 | .../caption_transformer_dac-12bit_cmm_crosssim.pt | DAC-12 CMM-CrossSim checkpoint |
| --coco-root | data/coco | 默认值 | data/coco | COCO 数据根目录 |
| --output | experiments/dac_range_root_cause/adaptive_dac12_metrics.json | 默认值 | experiments/dac_range_root_cause/adaptive_dac12_metrics.json | 指标 JSON 输出 |
| --device | cuda | 用户指定 (默认值) | cuda (if available) | 推理设备 |
| --limit | 1000 | 默认值 | 1000 | pycocoevalcap 图片数量 |
| --calibration-batches | 50 | 默认值 | 50 | layer-wise 校准 batch 数 |
| --batch-size | 32 | 默认值 | 32 | 生成 caption batch size |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |

## 实验结果

### 实验 1: Activation Clipping 统计（50 batch, 400 张图片）

**关键发现：Layer 0 self_attn Q/K/V 的 clip_rate 高达 85%，即 85% 的 activation 值被硬裁剪到 [-1, 1]**

| 层 | 组 | min | max | std | p99_abs | clip_rate |
|----|-----|-----|-----|-----|---------|-----------|
| layers.0.self_attn.q_proj | self_attn | -67.6 | 77.4 | 14.1 | 40.4 | **85.0%** |
| layers.0.self_attn.k_proj | self_attn | -67.6 | 77.4 | 14.1 | 40.4 | **85.0%** |
| layers.0.self_attn.v_proj | self_attn | -67.6 | 77.4 | 14.1 | 40.4 | **85.0%** |
| layers.0.self_attn.o_proj | self_attn | -42.5 | 51.6 | 6.2 | 19.0 | **82.9%** |
| layers.1.self_attn.q_proj | self_attn | -8.6 | 8.8 | 0.9 | 2.6 | 24.4% |
| layers.0.cross_attn.q_proj | cross_attn | -5.0 | 4.7 | 1.0 | 2.7 | 32.8% |
| layers.0.cross_attn.k_proj | cross_attn | -17.1 | 20.7 | 1.4 | 5.0 | 34.5% |
| layers.0.cross_attn.v_proj | cross_attn | -17.1 | 20.7 | 1.4 | 5.0 | 34.5% |
| layers.0.ffn.0 | ffn | -6.1 | 6.0 | 1.0 | 2.7 | 31.8% |
| layers.0.ffn.3 | ffn | 0.0 | 10.3 | 0.3 | 1.5 | **2.9%** |
| output_proj | output_proj | -15.4 | 16.0 | 2.0 | 5.5 | **59.7%** |

**规律**：
- Layer 0 self_attn 损伤最严重：Q/K/V 的激活范围高达 ±77，p99 约 40× 超出 [-1,1]
- Layer 1 明显更温和：经过 Layer 0 的 LayerNorm 后，范围降到 ±9，clip_rate 约 24%
- ffn.3 (ReLU 后) 最温和：clip_rate ~3%，但仍有 max~10 的值被裁剪
- Layer 0 和 Layer 1 的 cross_attn K/V 共享相同的 memory (encoder 输出)，clip_rate 一致 (~34%)

### 实验 2: 分布对比图

5 张分布图均确认：固定 [-1,1] 裁剪后，大量值堆积在 ±1 边界（图中黑色虚线），导致信息大量丢失；自适应量化（绿线）与原始分布（蓝线）高度重合。

典型表现：
- **self_attn.q_proj**: 原始分布在 ±80，[-1,1] 裁剪后尾部全部截断，分布严重畸变
- **ffn.0**: 范围 ±6，[-1,1] 裁剪后仍有 ~32% 值被裁，分布峰被削平
- **output_proj**: 范围 ±16，[-1,1] 裁剪后 ~60% 值丢失

### 实验 3: Adaptive DAC-12 三设置对照（1000 张图片）

| 设置 | BLEU-1 | BLEU-4 | CIDEr | SPICE | Δ BLEU-4 vs Baseline |
|------|--------|--------|-------|-------|----------------------|
| Baseline (数字) | 0.6879 | **0.2573** | 0.8970 | 0.1677 | reference |
| DAC-12 current [-1,1] | 0.5878 | **0.1686** | 0.5775 | 0.1230 | **-34.5%** |
| DAC-12 layer-wise adaptive | 0.6882 | **0.2566** | 0.8941 | 0.1670 | **-0.3%** |
| DAC-12 batch-wise adaptive | 0.6873 | **0.2574** | 0.8958 | 0.1672 | **+0.0%** |

### Layer-wise DAC Scales（校准得到，用于 layer-wise adaptive 设置）

| 层 | adaptive_absmax | 层 | adaptive_absmax |
|----|----------------|----|-----------------|
| layers.0.self_attn.q_proj | 77.4 | layers.1.self_attn.q_proj | 8.8 |
| layers.0.self_attn.k_proj | 77.4 | layers.1.self_attn.k_proj | 8.8 |
| layers.0.self_attn.v_proj | 77.4 | layers.1.self_attn.v_proj | 8.8 |
| layers.0.self_attn.o_proj | 51.6 | layers.1.self_attn.o_proj | 4.0 |
| layers.0.cross_attn.q_proj | 5.0 | layers.1.cross_attn.q_proj | 6.7 |
| layers.0.cross_attn.k_proj | 20.7 | layers.1.cross_attn.k_proj | 20.7 |
| layers.0.cross_attn.v_proj | 20.7 | layers.1.cross_attn.v_proj | 20.7 |
| layers.0.cross_attn.o_proj | 9.7 | layers.1.cross_attn.o_proj | 11.0 |
| layers.0.ffn.0 | 6.1 | layers.1.ffn.0 | 7.8 |
| layers.0.ffn.3 | 10.3 | layers.1.ffn.3 | 4.7 |
| output_proj | 16.0 | | |

## 结论

1. **Clipping 是主因，不是量化分辨率。** Layer 0 self_attn 的 85% activation 被 [-1,1] 裁剪，output_proj 的 60% 被裁剪。12/10/8 bit 形成同平台是因为 clipping 损伤远大于量化损伤。

2. **Adaptive DAC-12 完全恢复。** Layer-wise 和 batch-wise adaptive DAC-12 的 BLEU-4 分别回到 0.2566 和 0.2574，与 baseline 0.2573 差异仅 -0.3%/+0.0%，在统计噪声范围内。

3. **"DAC 是主导瓶颈"的结论需要修正。** 原结论观察到的是 CrossSim 默认输入范围设置的问题，并非 DAC 量化分辨率本身的问题。修正后表述：CrossSim QuantizerDAC 的固定 [-1, 1] 输入范围与 transformer activation 的动态范围严重不匹配，导致大量 clipping；通过 layer-wise 自适应范围可完全恢复。

4. **修复方案**：在 `build_crosssim_params()` 中启用 `core.mapping.inputs.mvm.percentile=1.0` 的 percentile 输入校准，使 CrossSim 在每次推理时动态自适应输入范围。Layer-wise 和 batch-wise 两种 adaptive 策略效果几乎相同，layer-wise 更适合部署（范围可离线预校准）。

## 备注

- 所有 activation 统计来源于 baseline 模型（量化前原始分布）
- ffn.3 (ReLU 输出) clip_rate 最低 (~3%)，因为 post-LayerNorm + ReLU 天然把范围约束在较小正值
- Layer 0 / Layer 1 cross_attn K/V 范围不同但 clip_rate 接近，因为它们共享相同的 encoder memory 输入
- Layer 1 self_attn 的 max/min (±9) 远小于 Layer 0 (±77)，原因是通过了 Layer 0 的 LayerNorm
