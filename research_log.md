# 研究日志

## 2026-05-21

### 当前进度

- [x] ResNet-50 图像编码器
- [x] Transformer 解码器（MultiHeadAttention + DecoderLayer）
- [x] CaptionTransformer 端到端模型
- [x] COCO 数据预处理管线
- [x] 训练脚本（AMP + 梯度累积 + checkpoint）
- [x] 基线模型训练完成（checkpoints/caption_transformer_epoch_10.pt）
- [x] 评估脚本（BLEU/METEOR/ROUGE）
- [x] 更换 CrossSim 替代 MemTorch
- [x] CrossSim decoder-only 映射验证（Logit MAE ~ 1e-6，几乎无损）
- [x] CrossSim write noise 全系列实验（it-10 ~ it-6）
- [x] CrossSim ADC 精度消融（adc-12 ~ adc-4）

### 基线模型（Ref-Transformer）

| 项目 | 配置 |
|------|------|
| 编码器 | ResNet-50 (fc 层前截断，2048-d 输出) |
| 解码器层数 | 2 |
| 隐藏维度 | 256 |
| 注意力头数 | 4 |
| FFN 维度 | 1024 |
| 最大序列长度 | 30 |
| 训练轮数 | 10 |
| 批次大小 | 24 |
| 学习率 | 1e-4 |
| 数据集 | MS COCO Captions 2014 |

| 指标 | 值 |
|------|-----|
| BLEU-1 | 0.6816 |
| BLEU-4 | 0.2464 |
| METEOR | 0.2276 |
| ROUGE-L | 0.4954 |
| CIDEr | 0.8546 |
| SPICE | 0.1680 |

---

## 2026-05-22

### 决策：更换模拟库 MemTorch → CrossSim

**背景**：经过一整天的消融实验，排除了 ADC resolution、tile_shape、3D 输入适配、output_proj 映射范围等因素。MemTorch 的 naive_map/naive_program/naive_scale + VTEAM 器件模型组合导致权重级系统性偏差（Logit MAE ~2.62），参数调优无法解决。

**决定**：放弃 MemTorch，改用 **CrossSim**（Sandia 国家实验室的 crossbar 仿真框架）。

### CrossSim Decoder-Only 映射结果

**首个 CrossSim decoder_only 评估**（`caption_transformer_crosssim_decoder.pt`）：

```text
Device: cuda
CrossSim scope: decoder_only
CrossSim tile shape: (128, 128)
CrossSim ADC/DAC: 0/0
CrossSim GPU: True
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CrossSim  loss: 2.534256, token_acc: 0.476609
Delta loss (crosssim - base): 0.000000
Delta acc  (crosssim - base): 0.000000
Logit MAE: 0.000001
Logit Max Error: 0.000072
Logit RMSE: 0.000001
```

**结论**：CrossSim decoder-only 映射几乎无损（Logit MAE ~ 1e-6，仅为浮点舍入误差级别）。

### CrossSim Write Noise消融

#### CrossSim Write Noise：it-6

**噪声 std=1e-2**（`crosssim_conditions/caption_transformer_it-6_crosssim.pt`）：

```text
Device: cuda
CrossSim scope: decoder_only
CrossSim tile shape: (128, 128)
CrossSim ADC/DAC: 0/0
CrossSim GPU: True
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CrossSim  loss: 5.824668, token_acc: 0.224745
Delta loss (crosssim - base): 3.290413
Delta acc  (crosssim - base): -0.251864
Logit MAE: 1.991278
Logit Max Error: 21.485548
Logit RMSE: 2.520760
```

| 条件 | Loss | Token Acc | Logit MAE | Logit Max Error | Logit RMSE |
| --- | --- | --- | --- | --- | --- |
| CrossSim 无噪声 | 2.534256 | 47.7% | 1e-6 | 7.2e-5 | 1e-6 |
| CrossSim it-6 (1e-2) | 5.824668 | 22.5% | 1.991 | 21.49 | 2.521 |

**分析**：

- it-6 write noise 导致 token 准确率从 47.7% 降至 22.5%，loss 从 2.53 升至 5.82
- MAE 从 ~0 升至 1.99，write noise 对模型输出有显著影响

#### CrossSim Write Noise：it-7

**噪声 std=1e-3**（`crosssim_conditions/caption_transformer_it-7_crosssim.pt`）：

```text
Device: cuda
CrossSim scope: decoder_only
CrossSim tile shape: (128, 128)
CrossSim ADC/DAC: 0/0
CrossSim GPU: True
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CrossSim  loss: 3.308090, token_acc: 0.392930
Delta loss (crosssim - base): 0.773834
Delta acc  (crosssim - base): -0.083679
Logit MAE: 1.063812
Logit Max Error: 13.412535
Logit RMSE: 1.392701
```

#### CrossSim Write Noise：it-8

**噪声 std=1e-4**（`crosssim_conditions/caption_transformer_it-8_crosssim.pt`）：

```text
Device: cuda
CrossSim scope: decoder_only
CrossSim tile shape: (128, 128)
CrossSim ADC/DAC: 0/0
CrossSim GPU: True
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CrossSim  loss: 2.534066, token_acc: 0.477382
Delta loss (crosssim - base): -0.000190
Delta acc  (crosssim - base): 0.000773
Logit MAE: 0.012248
Logit Max Error: 0.448982
Logit RMSE: 0.017217
```

#### CrossSim Write Noise：it-9

**噪声 std=1e-5**（`crosssim_conditions/caption_transformer_it-9_crosssim.pt`）：

```text
Device: cuda
CrossSim scope: decoder_only
CrossSim tile shape: (128, 128)
CrossSim ADC/DAC: 0/0
CrossSim GPU: True
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CrossSim  loss: 2.534284, token_acc: 0.476277
Delta loss (crosssim - base): 0.000028
Delta acc  (crosssim - base): -0.000331
Logit MAE: 0.001997
Logit Max Error: 0.066716
Logit RMSE: 0.002727
```

#### CrossSim Write Noise：it-10

**噪声 std=1e-6**（`crosssim_conditions/caption_transformer_it-10_crosssim.pt`）：

```text
Device: cuda
CrossSim scope: decoder_only
CrossSim tile shape: (128, 128)
CrossSim ADC/DAC: 0/0
CrossSim GPU: True
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CrossSim  loss: 2.534277, token_acc: 0.476443
Delta loss (crosssim - base): 0.000022
Delta acc  (crosssim - base): -0.000166
Logit MAE: 0.001048
Logit Max Error: 0.046411
Logit RMSE: 0.001462
```

#### Write Noise 退化曲线（最终汇总）

| 条件 | 噪声 std | Loss | Token Acc | Δ Acc | Logit MAE | Logit Max Error |
| --- | --- | --- | --- | --- | --- | --- |
| 无噪声 | 0 | 2.534 | 47.7% | — | ~0 | ~0 |
| it-10 | 1e-6 | 2.534 | 47.6% | -0.02pp | 0.001 | 0.05 |
| it-9 | 1e-5 | 2.534 | 47.6% | -0.03pp | 0.002 | 0.07 |
| it-8 | 1e-4 | 2.534 | 47.7% | +0.08pp | 0.012 | 0.45 |
| it-7 | 1e-3 | 3.308 | 39.3% | -8.4pp | 1.064 | 13.41 |
| it-6 | 1e-2 | 5.825 | 22.5% | -25.2pp | 1.991 | 21.49 |

**分析**：

- **安全区（it-10 ~ it-8）**：噪声 1e-6 ~ 1e-4，MAE ≤ 0.012，loss/acc 与基线无差异
- **退化临界点**：1e-4 → 1e-3，MAE 跳升 ~90×（0.012 → 1.06）
- **严重退化区（it-7 ~ it-6）**：噪声 1e-3 ~ 1e-2，MAE 1.0~2.0
- **it-8（1e-4）为推荐写入精度**：兼顾硬件可编程性与模型保真度

#### pycocoevalcap 全指标评估（limit=500）

| 条件 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.855 | 0.168 |
| baseline-crosssim | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.855 | 0.168 |
| **it-10 (1e-6)** | 0.6823 | 0.2472 | 0.2280 | 0.4962 | 0.856 | 0.168 |
| **it-9 (1e-5)** | 0.6823 | 0.2466 | 0.2278 | 0.4955 | 0.855 | 0.168 |
| **it-8 (1e-4)** | 0.6805 | **0.2502** | 0.2286 | 0.4970 | 0.857 | 0.168 |
| it-7 (1e-3) | 0.4647 | 0.0913 | 0.1171 | 0.3475 | 0.180 | 0.045 |
| it-6 (1e-2) | 0.1561 | **0.0000** | 0.0581 | 0.1993 | 0.008 | 0.001 |

> it-6 BLEU-4=0 原因：模型输出序列长度暴涨（testlen=10435 vs reflen=5983, ratio=1.74），大量生成无效 token。
> baseline-crosssim 与 baseline 完全一致，确认 CrossSim 零噪声映射无损。

**核心结论**：

- **it-8（1e-4）为最佳写入精度**：所有指标与基线持平甚至略优
- **安全区（it-10 ~ it-8）无需任何容错设计**
- **it-7（1e-3）为退化起点**：BLEU-4 从 0.25 暴跌至 0.09，METEOR 减半
- **it-6（1e-2）模型功能崩溃**：BLEU-4=0，SPICE≈0

### CrossSim ADC 分辨率消融

**数据来源**：`checkpoints/crosssim_adc_conditions/metrics_pycoco.json`

| 条件 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| baseline-crosssim | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| **adc-12** | 0.6770 | **0.2468** | 0.2282 | 0.4953 | 0.8552 | 0.1680 |
| **adc-10** | 0.6638 | 0.2396 | 0.2238 | 0.4893 | 0.8284 | 0.1649 |
| adc-8 | 0.6310 | 0.2166 | 0.2204 | 0.4785 | 0.7429 | 0.1561 |
| adc-6 | 0.2845 | 0.0265 | 0.1167 | 0.2785 | 0.0612 | 0.0659 |
| adc-4 | 0.0617 | ~0 | 0.0131 | 0.0951 | ~0 | 0.0 |

**分析**：

- **adc-12 几乎无损**：所有指标与 baseline 在统计噪声范围内一致
- **adc-10 轻微退化**：CIDEr 下降 ~3%（0.855→0.828），BLEU-4 下降 ~3%，属可接受范围
- **adc-8 明显退化**：CIDEr 下降 ~13%（0.855→0.743），各指标开始明显下滑
- **adc-6 严重退化**：BLEU-4 从 0.246 降至 0.027，模型基本失效
- **adc-4 功能崩溃**：BLEU-4 ≈ 0，SPICE = 0，CIDEr ≈ 0

**推荐 ADC ≥ 10-bit**：adc-10 退化 ~3% 可控；adc-8 退化 ~13% 需配合高写入精度。

### CrossSim DAC 分辨率消融

**数据来源**：`checkpoints/crosssim_dac_conditions/metrics_pycoco.json`

| 条件 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| baseline-crosssim | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| dac-12 | 0.5860 | 0.1637 | 0.1772 | 0.4349 | 0.5397 | 0.1242 |
| dac-10 | 0.5787 | 0.1602 | 0.1747 | 0.4336 | 0.5373 | 0.1236 |
| dac-8 | 0.5759 | 0.1583 | 0.1749 | 0.4338 | 0.5380 | 0.1224 |
| dac-6 | 0.5425 | 0.1456 | 0.1673 | 0.4255 | 0.5144 | 0.1187 |
| dac-4 | 0.2013 | 0.0411 | 0.0634 | 0.2537 | 0.1639 | 0.0571 |

**分析**：

- **DAC 整体影响远大于 ADC**：即使 dac-12 也使 BLEU-4 从 0.246 降至 0.164（-33%），而 adc-12 几乎无损
- **dac-12 ~ dac-8 呈平台区**：BLEU-4 ~0.16, CIDEr ~0.54，三者差异极小，说明此区间 DAC 精度提升收益递减
- **dac-6 略差于 dac-8**：CIDEr 再降 ~4%，但仍在同一量级
- **dac-4 严重崩溃**：BLEU-4 降至 0.041（-83%），SPICE 降至 0.057
- DAC 对性能的影响比 ADC 更关键：DAC 负责将模拟计算结果转回数字域供下一层使用，量化误差直接注入信号通路

**推荐 DAC ≥ 6-bit**：dac-8/10/12 几乎等价，不必追求高位；dac-4 不可用。

### CrossSim Read Noise 消融

**数据来源**：`checkpoints/crosssim_read_noise_conditions/metrics_pycoco.json`

| 条件 | read_noise_std | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| read-noise-0 | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| read-noise-1e-5 | 1e-5 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8548 | 0.1681 |
| read-noise-1e-4 | 1e-4 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8548 | 0.1681 |
| read-noise-1e-3 | 1e-3 | 0.6839 | **0.2506** | 0.2286 | 0.4964 | 0.8597 | 0.1687 |
| read-noise-1e-2 | 1e-2 | 0.6821 | **0.2505** | 0.2277 | 0.4972 | 0.8555 | 0.1663 |

**分析**：

- **Read noise 几乎无影响**：所有 read noise 条件（含 1e-2）的指标与 baseline 在统计噪声范围内一致
- **与 write noise 形成强烈对比**：write noise 在 1e-2 时模型完全崩溃（BLEU-4=0），而 read noise 在 1e-2 时 BLEU-4=0.2505 甚至略优于基线
- **1e-3 处有微弱正效应**：BLEU-4 0.2506 vs baseline 0.2464（+1.7%），CIDEr 0.8597 vs 0.8546，可能与 read noise 引入的微小扰动起到隐式正则化有关
- 物理直觉吻合：read noise 仅影响读取过程的瞬时值，不改变存储权重，多次读取可平均化；write noise 永久腐蚀权重，误差逐层累积

**结论**：Read noise 对 Transformer 图像描述模型基本无害，无需特殊容错设计。主要噪声威胁来自 write noise 和 DAC 量化。

### CrossSim Array Size 消融

**数据来源**：`checkpoints/crosssim_array_size_conditions/metrics_pycoco.json`

| 条件 | tile_shape | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-64×64 | 64×64 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-128×128 | 128×128 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-256×256 | 256×256 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| array-512×512 | 512×512 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |

**分析**：

- **Array size 64×64 ~ 512×512 全部与 baseline 完全一致**：在所有理想器件条件下，切 tile 后的矩阵-向量乘积在数学上等价，array size 不影响模型精度
- 该结果同时验证了 CrossSim 的 tile 切分/重组机制正确，无实现层面的误差
- Array size 的实际意义在于**与非理想效应的交互**：更小的 tile 意味着更多的 tile 边界、更多的 ADC/DAC 调用、更多的写噪声注入点。单独的 array size sweep 不足以暴露问题，需与 write noise + ADC/DAC 条件联合测试才能体现差异

---

## 下一步计划

1. [x] **DAC bit sweep** — 已完成（dac-12 ~ dac-4），DAC 影响远大于 ADC，推荐 ≥ 6-bit
1. [x] **Read noise sweep** — 已完成（std 0 ~ 1e-2），read noise 几乎无影响
1. [x] **Array size sweep** — 已完成（64×64 ~ 512×512），理想条件下无差异，需与非理想效应联合测试
1. **Module-wise sensitivity analysis** — 改变忆阻器交叉阵列规模（如 64×64 / 128×128 / 256×256 / 512×512），观察 array size 对模型保真度的影响
1. **Module-wise sensitivity analysis** — 分别对 attention 投影层（Q/K/V/O）和 FFN 层（FFN1/FFN2）施加噪声，定位对噪声最敏感的模块
1. **CMM-style mapping / CMM prototype** — 构建贴近 CMM 实际器件的映射参数集（非理想 Ron/Roff 分布、stuck-at-fault、IR drop 等），模拟真实 CMM 芯片行为
