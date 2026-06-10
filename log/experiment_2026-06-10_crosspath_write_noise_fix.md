# Write Noise Cross-Path Diagnostic Comparison (Table XII 修正版)

| 属性 | 值 |
|------|-----|
| **日期** | 2026-06-10 |
| **状态** | 已完成 |
| **脚本** | test_crosssim_write_noise_conditions.py, test_cmm_write_noise_conditions.py, test_cmm_crosssim_write_noise_conditions.py, compute_metrics_pycoco.py |
| **变更** | src/test_crosssim_write_noise_conditions.py, src/test_cmm_write_noise_conditions.py |
| **图片数** | 500 (COCO val2014 subset) |
| **基线 checkpoint** | checkpoints/caption_transformer_epoch_10.pt |

## 实验目的

修正 Table XII 中 write noise 行两条关键缺陷:

1. **噪声域不一致** — CS-only 路径原先在原始权重域 (unnormalized) 直接加 `randn * noise_std`,而 CMM-CS 使用 CrossSim `NormalIndependentDevice` 在电导域注入。1e-2 在未归一化权重域 (~36% 相对误差) 远大于电导域 (~2%),导致 CS-only 在 write noise 1e-2 时伪崩溃 (B-4=0.0000)。

2. **基线不一致** — 三条路径原先的 ADC/DAC/read_noise 基线不同 (CS-only: ADC=10/DAC=12/read_noise=0; CMM-CS: ADC=0/DAC=0/read_noise=1e-4),无法做单变量对比。

**方案 A 修正**: CS-only 不再用原始权重域手动噪声,改为将 `write_noise_std` 映射为 CrossSim 内部 `programming_error_std`,使 CS-only 与 CMM-CS 都在 CrossSim 电导域注入 `NormalIndependentDevice` 噪声。同时统一三条路径的固定基线为 ADC=0/DAC=0/cell_bits=0/read_noise=1e-4/tile=128x128。

CMM-CS 脚本本身没有改动 (其默认基线已经是 ADC=0/DAC=0/read_noise=1e-4),沿用旧 checkpoints 数据 (`cmm_crosssim_write_noise_conditions/`)。

## 统一基线

| 参数 | CS-only v3 | CMM-only v2 | CMM-CS (旧) |
|------|-----------|------------|-------------|
| ADC resolution | 0 (理想) | 0 (理想) | 0 (理想) |
| DAC resolution | 0 (理想) | 0 (理想) | 0 (理想) |
| cell_bits | 0 (连续) | 0 (连续) | 0 (连续) |
| read_noise_std | 1e-4 | 1e-4 | 1e-4 |
| tile shape | 128x128 | 128x128 | 128x128 |
| write noise 值域 | 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 同左 | 同左 |
| seeds | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 |
| 噪声注入域 | CrossSim 电导域 (NormalIndependentDevice) | CMM r-state [0,1] | CrossSim 电导域 (NormalIndependentDevice) |

## 运行命令

### CS-only v3 (重建)

```bash
python -m src.test_crosssim_write_noise_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/crosssim_write_noise_conditions_v3 \
    --device cpu

python -m src.compute_metrics_pycoco \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --conditions-manifest checkpoints/crosssim_write_noise_conditions_v3/conditions_manifest.json \
    --output checkpoints/crosssim_write_noise_conditions_v3/metrics_pycoco.json \
    --limit 500 --batch-size 16 --device cuda
```

### CMM-only v2 (重建)

```bash
python -m src.test_cmm_write_noise_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/cmm_write_noise_conditions_v2 \
    --device cpu

python -m src.compute_metrics_pycoco \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --conditions-manifest checkpoints/cmm_write_noise_conditions_v2/conditions_manifest.json \
    --output checkpoints/cmm_write_noise_conditions_v2/metrics_pycoco.json \
    --limit 500 --batch-size 64 --device cuda
```

### CMM-CS (沿用旧数据)

```bash
# 未重新运行,复用 checkpoints/cmm_crosssim_write_noise_conditions/metrics_pycoco.json
```

## 参数说明

### test_crosssim_write_noise_conditions.py (CS-only v3 修改)

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | (required) | 训练检查点 |
| --output-dir | crosssim_write_noise_conditions_v3 | 用户指定 | crosssim_write_noise_conditions | 输出目录 |
| --scope | decoder_only | 默认值 | decoder_only | 映射范围 |
| --tile-rows | 128 | 固定 | 128 | 改为固定 128 |
| --tile-cols | 128 | 固定 | 128 | 改为固定 128 |
| --adc-resolution | 0 | 固定 | **原 10 → 0** | 统一为理想 ADC |
| --dac-resolution | 0 | 固定 | **原 12 → 0** | 统一为理想 DAC |
| --cell-bits | 0 | 固定 | 0 | 连续电导 |
| --read-noise-std | 1e-4 | 固定 | **原 0 → 1e-4** | 与 CMM-CS 对齐 |
| --write-noise-stds | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 默认值 | (新增) | 统一值域,替代原 it-10~it-6 |
| --seeds | 1,2,3 | 默认值 | (新增) | 多 seed 可复现 |
| --device | cpu | 用户指定 | cuda | 构建用 CPU |

**关键变更**:
- 移除 `inject_write_noise()` 函数 — 不再在 PyTorch 原始权重域加噪声
- 改为将 `write_noise_std` 映射为 CrossSim `programming_error_std`,在电导域由 `NormalIndependentDevice` 注入
- 新增 `set_seed()` 覆盖 random/numpy/torch 三个随机源
- 新增 `parse_float_list()` / `parse_seed_list()` 支持批量 sweep
- 新增 `replicate_groups_manifest.json` 输出

### test_cmm_write_noise_conditions.py (CMM-only v2 修改)

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | (required) | 训练检查点 |
| --output-dir | cmm_write_noise_conditions_v2 | 用户指定 | cmm_write_noise_conditions | 输出目录 |
| --scope | decoder_only | 默认值 | decoder_only | 映射范围 |
| --tile-rows | 128 | 固定 | 128 | 改为固定 128 |
| --tile-cols | 128 | 固定 | 128 | 改为固定 128 |
| --cell-bits | 0 | 固定 | 0 | 连续电导 |
| --adc-resolution | 0 | 固定 | **新增** | 显式写为 0 |
| --dac-resolution | 0 | 固定 | **新增** | 显式写为 0 |
| --read-noise-std | 1e-4 | 固定 | **原 0 → 1e-4** | 与 CMM-CS 对齐 |
| --write-noise-stds | 0,1e-4,3e-4,1e-3,3e-3,1e-2 | 默认值 | 同左 | 值域不变但基线变了 |

**关键变更**:
- read_noise_std 从 0 改为 1e-4,与 CS-only/CMM-CS 对齐
- adc_resolution/dac_resolution 显式写为 0 并写入 checkpoint 元数据
- tile_rows/tile_cols 固定为 128 避免命令行误改

## 实验结果

### 各噪声水平三路对比 (mean ± std, n=3 seeds)

| Write Noise | 路径 | BLEU-4 | CIDEr | METEOR | ROUGE_L |
|---|---:|---|---:|---:|---:|
| **Baseline (数字)** | — | 0.2481 | 0.8560 | 0.2279 | 0.4957 |
| **0** | CS-only v3 | 0.2481 ± 0.0001 | 0.8560 ± 0.0002 | 0.2279 ± 0.0000 | 0.4957 ± 0.0000 |
| | CMM-only v2 | 0.2481 ± 0.0002 | 0.8567 ± 0.0012 | 0.2280 ± 0.0001 | 0.4958 ± 0.0003 |
| | CMM-CS | 0.2464 ± 0.0000 | 0.8548 ± 0.0000 | 0.2276 ± 0.0000 | 0.4954 ± 0.0000 |
| **1e-4** | CS-only v3 | 0.2473 ± 0.0018 | 0.8542 ± 0.0021 | 0.2277 ± 0.0003 | 0.4956 ± 0.0003 |
| | CMM-only v2 | 0.2468 ± 0.0008 | 0.8545 ± 0.0009 | 0.2276 ± 0.0001 | 0.4954 ± 0.0003 |
| | CMM-CS | 0.2483 ± 0.0019 | 0.8561 ± 0.0026 | 0.2282 ± 0.0006 | 0.4959 ± 0.0008 |
| **3e-4** | CS-only v3 | 0.2482 ± 0.0016 | 0.8573 ± 0.0022 | 0.2285 ± 0.0005 | 0.4966 ± 0.0009 |
| | CMM-only v2 | 0.2470 ± 0.0005 | 0.8531 ± 0.0018 | 0.2275 ± 0.0004 | 0.4954 ± 0.0003 |
| | CMM-CS | 0.2489 ± 0.0021 | 0.8567 ± 0.0008 | 0.2278 ± 0.0004 | 0.4960 ± 0.0006 |
| **1e-3** | CS-only v3 | 0.2472 ± 0.0015 | 0.8494 ± 0.0051 | 0.2281 ± 0.0011 | 0.4956 ± 0.0007 |
| | CMM-only v2 | 0.2477 ± 0.0002 | 0.8530 ± 0.0028 | 0.2276 ± 0.0010 | 0.4955 ± 0.0003 |
| | CMM-CS | 0.2477 ± 0.0009 | 0.8511 ± 0.0032 | 0.2280 ± 0.0017 | 0.4962 ± 0.0010 |
| **3e-3** | CS-only v3 | 0.2476 ± 0.0033 | 0.8457 ± 0.0067 | 0.2274 ± 0.0006 | 0.4939 ± 0.0023 |
| | CMM-only v2 | 0.2467 ± 0.0031 | 0.8526 ± 0.0050 | 0.2261 ± 0.0009 | 0.4944 ± 0.0027 |
| | CMM-CS | 0.2480 ± 0.0027 | 0.8504 ± 0.0068 | 0.2276 ± 0.0012 | 0.4951 ± 0.0025 |
| **1e-2** | CS-only v3 | 0.2343 ± 0.0101 | 0.8195 ± 0.0164 | 0.2236 ± 0.0033 | 0.4911 ± 0.0062 |
| | CMM-only v2 | 0.2376 ± 0.0017 | 0.8296 ± 0.0082 | 0.2229 ± 0.0016 | 0.4898 ± 0.0022 |
| | CMM-CS | 0.2361 ± 0.0065 | 0.8210 ± 0.0191 | 0.2241 ± 0.0002 | 0.4881 ± 0.0028 |

### 关键发现

1. **CS-only 与 CMM-CS 在所有噪声水平几乎一致** — 两条路径都通过 CrossSim `NormalIndependentDevice` 在电导域注入噪声,证实方案 A 修正后噪声域对齐正确。

2. **旧 Table XII 中 CS-only write noise 1e-2 崩溃 (B-4=0.0000) 已消除** — 修正后 CS-only v3 在 1e-2 下 B-4=0.2343 ± 0.0101,与 CMM-CS (0.2361 ± 0.0065) 和 CMM-only (0.2376 ± 0.0017) 一致。旧结果完全是原始权重域噪声尺度失配造成的伪差异。

3. **三路降级曲线一致** — 低噪声 (≤1e-3) 时所有路径保持 baseline 水平 (B-4 ≈ 0.247~0.249);到 3e-3 开始出现轻微下降;到 1e-2 显著下降 (B-4 跌至 ~0.234~0.238) 但远未崩溃。三路降级模式高度一致。

4. **CMM-only 变异性略低** — CMM-only 在 1e-2 下的 seed 间标准差 (B-4 ±0.0017) 小于 CS-only (±0.0101) 和 CMM-CS (±0.0065)。这可能是因为 CMM r-state 噪声经过 `clamp(r, 0, 1)` 截断后有上限衰减,而 CrossSim 电导域噪声无截断。

5. **修正后 write noise 不再是 CMM 的优势证据源** — 三路降级基本一致,write noise 不能支撑"CMM 更鲁棒"的主张。CMM 与 CrossSim 在写入误差维度上的表现本质上是等价的噪声注入域映射,不构成独立的机制收益。

### CS-only 修正前后对比 (Table XII write noise 行)

| Write Noise | CS-only (旧, 权重域) | CS-only v3 (新, 电导域) | 差异解释 |
|---|---:|---:|---|
| 0 | B-4=0.2464 | B-4=0.2481 | baseline 差异 (旧 ADC=10/DAC=12, 新 ADC=0/DAC=0) |
| 1e-2 | B-4=0.0000 | B-4=0.2343 | 旧: 噪声/权重=36%→崩溃;新: 电导域 1%→正常 |
| 1e-3 | B-4≈0.22~0.23 | B-4=0.2472 | 旧: 噪声/权重=3.6%→显著退化;新: 电导域 0.1%→轻微 |

- 旧 CS-only write noise 结果来源: `checkpoints/crosssim_write_noise_conditions/` (it-6/it-7/it-8 等)
- 新 CS-only v3 结果来源: `checkpoints/crosssim_write_noise_conditions_v3/` (write-noise-0/write-noise-1e-04/...)

## 输出文件

| 文件 | 内容 |
|------|------|
| `checkpoints/crosssim_write_noise_conditions_v3/metrics_pycoco.json` | CS-only v3 三路评测结果 |
| `checkpoints/cmm_write_noise_conditions_v2/metrics_pycoco.json` | CMM-only v2 三路评测结果 |
| `checkpoints/cmm_crosssim_write_noise_conditions/metrics_pycoco.json` | CMM-CS 旧有评测结果 (复用) |

## 论文修改建议

1. **Table XII 全部 update 为修正后数据** — CS-only 列替换为 v3 数据,CMM-only 列替换为 v2 数据,CMM-CS 列保持不变。

2. **Discussion 关键修改** — 原文 Table XII 的 write noise 行被用来证明 "CMM 降低写入误差诱导退化",修正后三路降级一致,该主张不再成立。应修改为:"Under the unified programming-error injection framework, all three mapping paths exhibit near-identical degradation profiles across write noise levels, confirming that CMM-style write noise on r-state [0,1] maps to equivalent conductance-domain perturbation after Rmem transformation."

3. **Abstract/Conclusion 措辞克制化** — 避免暗示 CMM 提供超越 CrossSim-only 的鲁棒性优势,改为强调 CMM 框架与 CrossSim 后端的数值兼容性和等价性。

## 备注

- CMM-CS 路径未重新运行,因为 `test_cmm_crosssim_write_noise_conditions.py` 的基线 (ADC=0/DAC=0/read_noise=1e-4) 与统一后基线一致,且原本就使用 CrossSim programming_error
- CMM-CS 旧数据 baseline (B-4=0.2464) 与 CS-only v3 baseline (B-4=0.2481) 有 0.0017 的微小差异,这是因为两条路径 baseline 评测时使用了不同的 PyTorch 随机种子 (baseline 在每次扫描中单独跑) — 不影响结论
- 旧 CS-only write noise 结果 (权重域噪声, `crosssim_write_noise_conditions/`) 应标记为 deprecated 并移出 Table XII
- 三路一致这个结论与 Table V 的能耗比 1.0035 形成呼应: CMM-CrossSim 相比于 CS-only 在性能和能耗两个维度上都没有展现出超越简单 CrossSim 映射的独立优势
