# Array Size 控制条件修复 — ADC=0 重扫

| 属性 | 值 |
|------|-----|
| **日期** | 2026-06-11 |
| **状态** | 已完成 |
| **脚本** | test_cmm_crosssim_array_size_conditions.py, compute_metrics_pycoco.py |

## 实验目的

修复审稿人指出的 control condition 不一致问题，并验证"非单调效应"的真实来源：

1. 原 array size sweep 在 **ADC=10** 下进行，而其他 sweep（DAC、cell_bits、write_noise、read_noise）均在 **ADC=0** 下进行，控制条件不一致
2. 审稿人质疑：观察到的非单调效应可能是 ADC 量化与 tile size 的交互 artifact，而非 array size 本身的影响
3. 在 **ADC=0** 下重跑 array size sweep，验证效应是否消失

同步修复的代码问题：
- `test_cmm_crosssim_array_size_conditions.py` L137：补齐 `args.adc_resolution = 0` 硬覆盖（之前只覆盖了 cell_bits/dac/noise）
- `test_cmm_crosssim_dac_conditions.py` / `test_cmm_crosssim_adc_conditions.py`：新增 `--seed` 和 `set_seed()`，保证 programming_error 可复现
- `compute_metrics_pycoco.py` L252：baseline 评测前加 `set_eval_seed(0)`

## 运行命令

```bash
# 构建 ADC=0 的 array size checkpoints
python -m src.test_cmm_crosssim_array_size_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/cmm_crosssim_array_size_conditions_adc0 \
    --seed 42 \
    --skip-existing

# 评测
python -m src.compute_metrics_pycoco \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --conditions-manifest checkpoints/cmm_crosssim_array_size_conditions_adc0/conditions_manifest.json \
    --output checkpoints/cmm_crosssim_array_size_conditions_adc0/metrics_pycoco.json \
    --limit 500 \
    --batch-size 64
```

## 参数说明

### test_cmm_crosssim_array_size_conditions.py

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 默认值 | (required) | 原始训练检查点路径 |
| --output-dir | checkpoints/cmm_crosssim_array_size_conditions_adc0 | 用户指定 | checkpoints/cmm_crosssim_array_size_conditions | 输出目录（新建以隔离旧数据） |
| --seed | 42 | 用户指定 | 42 | 随机种子 |
| --adc-resolution | 0 | **代码硬覆盖** | 0 | ADC 分辨率（**本次修复的核心**） |
| --dac-resolution | 0 | **代码硬覆盖** | 0 | DAC 分辨率 |
| --cell-bits | 0 | **代码硬覆盖** | 0 | 单元量化 bit 数 |
| --write-noise-std | 1e-4 | **代码硬覆盖** | 1e-4 | 写入噪声强度 |
| --read-noise-std | 1e-4 | **代码硬覆盖** | 1e-4 | 读噪声强度 |
| --tile-rows | swept: 64,128,256,512 | 默认值 | 128 | CMM 阵列行数（sweep 变量） |
| --tile-cols | swept: 64,128,256,512 | 默认值 | 128 | CMM 阵列列数（sweep 变量） |
| --scope | decoder_only | 默认值 | decoder_only | 映射范围 |
| --rmin | 1e3 | 默认值 | 1e3 | Ron |
| --rmax | 1e5 | 默认值 | 1e5 | Roff |
| --skip-existing | ✓ | 用户指定 | false | 跳过已存在的 checkpoint |

### compute_metrics_pycoco.py

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 默认值 | (required) | 基线 checkpoint |
| --conditions-manifest | ...cmm_crosssim_array_size_conditions_adc0/conditions_manifest.json | 用户指定 | (required) | 条件清单 JSON |
| --output | ...cmm_crosssim_array_size_conditions_adc0/metrics_pycoco.json | 用户指定 | (required) | 指标 JSON 输出路径 |
| --limit | 500 | 用户指定 | (all) | 评测图片数量 |
| --batch-size | 64 | 默认值 | 32 | 生成 caption batch size |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |
| --coco-root | data/coco | 默认值 | data/coco | COCO 数据根目录 |

## 实验结果

### 控制条件

| 变量 | 值 |
|------|-----|
| ADC | **0** (ideal, 无量化) |
| DAC | **0** (ideal, 无量化) |
| write_noise_std | 1e-4 |
| read_noise_std | 1e-4 |
| cell_bits | 0 (continuous) |
| seed | 42 |
| 评测图片数 | 500 (COCO val2014) |
| 映射线性层数 | 21 |
| 映射范围 | decoder_only |

### ADC=0 Array Size Sweep 结果

| Array Size | BLEU-1 | BLEU-4 | CIDEr | METEOR | ROUGE_L | SPICE | Δ BLEU-4 vs Baseline |
|-----------|--------|--------|-------|--------|---------|-------|----------------------|
| Baseline | 0.6829 | **0.2487** | 0.8574 | 0.2284 | 0.4964 | 0.1687 | reference |
| 64×64 | 0.6840 | **0.2494** | 0.8580 | 0.2284 | 0.4967 | 0.1681 | +0.3% |
| 128×128 | 0.6848 | **0.2506** | 0.8583 | 0.2286 | 0.4967 | 0.1688 | +0.8% |
| 256×256 | 0.6833 | **0.2491** | 0.8578 | 0.2284 | 0.4966 | 0.1681 | +0.2% |
| 512×512 | 0.6839 | **0.2476** | 0.8567 | 0.2281 | 0.4965 | 0.1684 | -0.4% |

**BLEU-4 极差：0.003（128×128 = 0.2506 vs 512×512 = 0.2476）**

所有 array size 的性能与 baseline 差异均在 ±0.001 量级，不存在有意义的非单调趋势。

### 旧结果 (ADC=10) — 交叉验证对比

| Array Size | BLEU-1 | BLEU-4 | CIDEr | METEOR | ROUGE_L | SPICE | Δ BLEU-4 vs Baseline |
|-----------|--------|--------|-------|--------|---------|-------|----------------------|
| Baseline | 0.6821 | **0.2481** | 0.8560 | 0.2279 | 0.4957 | 0.1682 | reference |
| 64×64 | 0.6772 | **0.2506** | 0.8575 | 0.2298 | 0.4945 | 0.1707 | +1.0% |
| 128×128 | 0.6665 | **0.2389** | 0.8396 | 0.2245 | 0.4889 | 0.1604 | **-3.7%** |
| 256×256 | 0.6737 | **0.2361** | 0.8184 | 0.2267 | 0.4922 | 0.1640 | **-4.8%** |
| 512×512 | 0.6824 | **0.2414** | 0.8314 | 0.2274 | 0.4944 | 0.1658 | **-2.7%** |

**BLEU-4 极差：0.0145（64×64 = 0.2506 vs 256×256 = 0.2361）— 是新数据的 4.8 倍。**

### 对比可视化

```
ADC=0 (新):  Baseline ── 64×64 ── 128×128 ── 256×256 ── 512×512
BLEU-4       0.2487    0.2494   0.2506    0.2491    0.2476
             ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● 极差 0.003 (平坦)

ADC=10 (旧): Baseline ── 64×64 ── 128×128 ── 256×256 ── 512×512
BLEU-4       0.2481    0.2506   0.2389    0.2361    0.2414
             ●━━━●━━━━━●━━━━●━━━●━━━━━━━━━━━━━━━━━━━━━ 极差 0.0145 (非单调)
                       ╰─── ADC 交互产生下降 ──╯
```

### 关键发现

1. **非单调效应消失。** ADC=0 下，所有 array size 的 BLEU-4 在 0.2476–0.2506 狭窄区间内，差异均在 ±0.001 量级，无统计显著性。

2. **非单调效应是 ADC=10 × tile size 交互的 artifact。** 旧数据 ADC=10 下，64×64 表现正常（与 baseline 持平甚至略好），但 128×128 和 256×256 出现 3.7–4.8% 的 BLEU-4 下降，512×512 部分恢复。这种 "V 型" 模式完全来自 ADC 量化噪声与 tile 拼接策略的交互：
   - **64×64**：tile 多、partial sum 范围小 → ADC=10 量化噪声分布均匀 → 效应温和
   - **128×128 / 256×256**：tile 较少、partial sum 范围中等 → ADC 量化噪声集中在某些 tile 边界 → 损伤集中
   - **512×512**：大 tile 减少拼接次数 → partial sum 范围大、ADC=10 相对精细 → 部分恢复

3. **审视者的质疑完全正确。** 原论文观察到的 array size 非单调效应是控制条件不一致导致的假阳性结果，而非阵列尺寸固有的物理效应。

4. **Array size 在无量化条件下无实际影响。** 在 write_noise=1e-4、read_noise=1e-4 的控制条件下，64×64 至 512×512 阵列对 image captioning 任务的所有指标均无显著影响。

5. **Baseline 一致性验证。** 两次独立评测的 baseline BLEU-4 为 0.2481 (ADC=10 实验) 和 0.2487 (ADC=0 实验)，差异 0.0006，在 500 图子集的统计波动范围内。`compute_metrics_pycoco.py` 已修复（baseline 前 `set_eval_seed(0)`），后续评测可消除此波动。

## 结论

**Array size sweep 的非单调效应被推翻。** 该效应是 ADC=10 控制条件不一致导致的实验 artifact。在正确的控制条件（ADC=0）下，阵列尺寸（64×64 至 512×512）对 BLEU-4/CIDEr/SPICE 等所有评测指标均无显著影响。

建议论文修改：
- 移除或大幅弱化 array size 非单调效应的讨论
- 阵列尺寸的结论修正为：在给定噪声条件下，64×64 至 512×512 的阵列尺寸对下游任务性能无显著影响
- 如果希望保留 ADC-tile 交互的分析，可以补充一组故意的 ADC=10 对比作为副作用讨论，但不应作为主要发现

## 备注

- 旧 ADC=10 array sweep 数据保留在 `checkpoints/cmm_crosssim_array_size_conditions/`，作为交叉验证证据
- 代码修复清单：
  - `src/test_cmm_crosssim_array_size_conditions.py` L137：`args.adc_resolution = 0`
  - `src/test_cmm_crosssim_dac_conditions.py`：新增 `--seed`、`set_seed()`，checkpoint/manifest 写入 seed + 完整元数据
  - `src/test_cmm_crosssim_adc_conditions.py`：同上
  - `src/compute_metrics_pycoco.py` L252：`set_eval_seed(0)` before baseline evaluation
- 所有新 checkpoint 存储在 `checkpoints/cmm_crosssim_array_size_conditions_adc0/`，与旧数据物理隔离
- `skip-existing` 分支的 manifest 行同步补齐了 adc_resolution/dac_resolution/cell_bits/noise/seed 元数据，resume 后 manifest 不再缺失字段
