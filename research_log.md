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

## 2026-05-25

### CMM 论文式映射接入

**新增文件**：`src/cmm.py`、`src/map_cmm.py`、`src/check_cmm_linear_equivalence.py`

**CMMLinear 设计**（`cmm.py`）：

- 内部状态 `r_pos`/`r_neg`（归一化忆阻值 0~1），替代 CrossSim 的 VTEAM 器件电导模型
- Forward: `Rmem = Ron*r + Roff*(1-r)` → 归一化有效权重 `(Rmem - Rmin)/(Rmax - Rmin)` → tile 分组 partial sum
- `from_linear()`: `nn.Linear.weight` → 正/负归一化 → `r = 1 - w` → 可选写入噪声（Gaussian on `r`）→ clamp [0,1]
- 读噪声：Gaussian 比例扰动 `Rmem` → clamp `[Rmin, Rmax]` → 转为归一化权重
- `cell_bits` 支持 weight state 量化（0 = 连续）；Bias 保留在数字域
- Tile 分组前向按 `tile_rows × tile_cols` 分块

**map_cmm.py**：

- 复用 CrossSim mapper 的 scope 机制（`output_only`/`layers_only`/`decoder_only`）
- 参数：`--rmin`（1e3）、`--rmax`（1e5）、`--cell-bits`（0）、`--write-noise-std`（0）、`--read-noise-std`（0）
- Checkpoint: `cmm_model_state_dict` + `cmm_args`

| 特性 | CrossSim AnalogLinear | CMM CMMLinear |
| --- | --- | --- |
| 器件模型 | VTEAM 物理模型 | 论文式 r 状态 |
| 权重映射 | naive_map + naive_program | 正/负归一化 + r = 1-w |
| 写噪声 | programming_error (电导) | Gaussian on r |
| 读噪声 | read_noise (电导) | Gaussian on Rmem |
| ADC/DAC | 内置量化 | 暂无（tile 结构已预留） |

### CMM 等价性验证

**测试命令**：`python -m src.check_cmm_linear_equivalence`

```text
Device: cpu
Tile shape: (3, 5)
2D MAE: 0.0000000894
2D Max error: 0.0000003576
3D MAE: 0.0000000484
3D Max error: 0.0000002384

=== CMM Mapping Scope Check ===
Expected decoder_only linear layers: 21
Mapped CMMLinear layers: 21
Encoder CMMLinear layers: 0
```

**结论**：

- 理想条件下 `CMMLinear` ≡ `nn.Linear`：2D/3D MAE ~ 1e-7，Max error ~ 3e-7，纯浮点舍入误差级别
- Scope 正确：21 个 decoder 线性层全部映射，0 个 encoder 层被误映射
- 等价性验证通过，可以进入下一步 decoder_only 替换正确性评估

### CMM Decoder-Only 评估

**测试命令**：`python -m src.evaluate_crosssim --checkpoint ... --crosssim-checkpoint checkpoints/caption_transformer_cmm.pt`

```text
Device: cuda
CMM format: cmm_v1
CMMLinear layers: 21
CMM scope: decoder_only
CMM tile shape: (128, 128)
CMM Rmin/Rmax: 1000.0/100000.0
CMM write/read noise std: 0.0/0.0
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
CMM  loss: 2.534256, token_acc: 0.476609
Delta loss (cmm - base): 0.000000
Delta acc  (cmm - base): 0.000000
Logit MAE: 0.000001
Logit Max Error: 0.000055
Logit RMSE: 0.000001
```

**结论**：

- CMM decoder_only 映射完全无损：loss/acc 与 baseline 三位小数点一致，Logit MAE ~ 1e-6
- 21 个 CMMLinear 层正确加载，scope/rmin/rmax/noise 元数据完整
- PyTorch CMM 等效仿真管线全部验证通过（等价性 + decoder_only + 评估指标）

### CMM cell_bits 量化消融

**数据来源**：`checkpoints/cmm_cell_bits_conditions/metrics_pycoco.json`

| 条件 | cell_bits | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| cell-continuous | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| **cell-8bit** | 8 | 0.6783 | 0.2455 | 0.2280 | 0.4951 | 0.8503 | 0.1668 |
| **cell-6bit** | 6 | 0.6721 | 0.2443 | 0.2254 | 0.4902 | 0.8377 | 0.1643 |
| **cell-4bit** | 4 | 0.6528 | 0.2266 | 0.2243 | 0.4892 | 0.8018 | 0.1623 |
| cell-3bit | 3 | 0.5621 | 0.1708 | 0.1974 | 0.4239 | 0.6624 | 0.1418 |
| cell-2bit | 2 | 0.0120 | ~0 | 0.0140 | 0.0175 | ~0 | 0.0145 |

**分析**：

- **cell-continuous 与 baseline 完全一致**：连续状态（cell_bits=0）下 CMM 映射无损，等价性验证通过
- **cell-8bit 几乎无损**：BLEU-4 0.2455 vs 0.2464（-0.4%），CIDEr 0.8503 vs 0.8546（-0.5%），8-bit 量化足以保持模型精度
- **cell-6bit 轻微退化**：CIDEr 下降 ~2%（0.855→0.838），BLEU-4 下降 ~0.9%，整体可控
- **cell-4bit 明显退化**：CIDEr 下降 ~6%（0.855→0.802），BLEU-4 下降 ~8%，各指标开始显著下滑
- **cell-3bit 严重退化**：BLEU-4 从 0.246 降至 0.171（-30%），CIDEr 下降 ~22%，模型质量大幅受损
- **cell-2bit 功能崩溃**：BLEU-4 ≈ 0，CIDEr ≈ 0，2-bit 量化无法保留有效权重信息

**核心结论**：
- **推荐 cell_bits ≥ 6**：6-bit 退化 ~2% 可控；8-bit 基本保持全精度
- **cell_bits=4 为退化起点**：各指标下滑明显但尚可接受
- 与 CrossSim ADC 消融对比：CMM cell_bits 量化影响介于 ADC-6（崩溃）与 ADC-8（明显退化）之间，但崩溃阈值更高（cell-2bit 才崩溃 vs ADC-4 崩溃）
- PyTorch CMM 等效仿真管线（等价性 → decoder_only → caption 指标）全部验证通过，可以进入 CrossSim 接入阶段

### CMM ADC 分辨率消融

**数据来源**：`checkpoints/cmm_adc_conditions/metrics_pycoco.json`

| 条件 | adc_resolution | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| adc-ideal | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| **adc-12bit** | 12 | 0.6835 | 0.2497 | 0.2288 | 0.4968 | 0.8603 | 0.1691 |
| **adc-10bit** | 10 | 0.6851 | 0.2521 | 0.2291 | 0.4980 | 0.8597 | 0.1679 |
| **adc-8bit** | 8 | 0.6818 | 0.2491 | 0.2293 | 0.4978 | 0.8517 | 0.1671 |
| adc-6bit | 6 | 0.6774 | 0.2463 | 0.2258 | 0.4955 | 0.8357 | 0.1655 |
| adc-4bit | 4 | 0.6539 | 0.2201 | 0.2214 | 0.4756 | 0.7830 | 0.1608 |

**分析**：

- **adc-ideal 与 baseline 完全一致**：说明 CMM 的理想 ADC 路径无额外数值误差
- **adc-12 / adc-10 / adc-8 均未出现退化**：BLEU-4 与 CIDEr 均与 baseline 持平甚至略优，说明在当前 CMM 仿真中，8-bit 以上 ADC 对 caption 质量几乎无影响
- **adc-6 开始出现轻微退化**：CIDEr 从 0.8546 降至 0.8357（-2.2%），BLEU-4 基本持平，属于边缘可接受区间
- **adc-4 明显退化**：BLEU-4 从 0.2464 降至 0.2201（-10.7%），CIDEr 从 0.8546 降至 0.7830（-8.4%），但仍未出现 CrossSim adc-4 那种完全崩溃
- **CMM ADC 敏感性显著弱于 CrossSim**：CrossSim 在 adc-6 已严重退化、adc-4 基本崩溃；而 CMM 在 adc-6 仅轻微退化、adc-4 仍保持可用 caption 质量
- 这说明当前 CMM 的 ADC 量化路径比 CrossSim/VTEAM 模型更平滑，误差传播更弱；后续需要在 CrossSim 接入阶段验证这是否来自 CMM 等效模型本身的“理想化”

**核心结论**：

- **推荐 CMM ADC ≥ 6-bit**：adc-6 退化约 2%，可接受；adc-8 及以上基本无损
- **adc-4 为明显退化起点**，但远未达到 CrossSim 同级别崩溃程度
- 当前 CMM 中，ADC 不是主要瓶颈；其影响弱于 `cell_bits`，也弱于 CrossSim 中 ADC 的影响

### CMM DAC 分辨率消融

**数据来源**：`checkpoints/cmm_dac_conditions/metrics_pycoco.json`

| 条件 | dac_resolution | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| dac-ideal | 0 | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| **dac-12bit** | 12 | 0.6815 | 0.2478 | 0.2278 | 0.4955 | 0.8557 | 0.1682 |
| **dac-10bit** | 10 | 0.6835 | 0.2496 | 0.2287 | 0.4963 | 0.8605 | 0.1688 |
| **dac-8bit** | 8 | 0.6811 | 0.2462 | 0.2275 | 0.4953 | 0.8523 | 0.1662 |
| dac-6bit | 6 | 0.6740 | 0.2472 | 0.2275 | 0.4935 | 0.8536 | 0.1684 |
| dac-4bit | 4 | 0.6702 | 0.2309 | 0.2232 | 0.4846 | 0.8107 | 0.1606 |

**分析**：

- **dac-ideal 与 baseline 完全一致**：说明 CMM 的理想 DAC 路径无额外数值误差
- **dac-12 / dac-10 / dac-8 基本无损**：各项指标与 baseline 持平，甚至 dac-10 略优（CIDEr 0.8605 vs 0.8546），说明在当前 CMM 仿真中，8-bit 以上 DAC 对 caption 质量几乎无影响
- **dac-6 仍保持稳定**：BLEU-4 0.2472 与 baseline 基本一致，CIDEr 0.8536 仅下降 0.1%，说明 CMM 对 DAC 量化同样不敏感
- **dac-4 才开始明显退化**：BLEU-4 从 0.2464 降至 0.2309（-6.3%），CIDEr 从 0.8546 降至 0.8107（-5.1%），但仍远好于 CrossSim dac-4 的严重崩溃表现
- **CMM DAC 敏感性远弱于 CrossSim**：CrossSim 在 dac-12 就已有明显性能损失，而 CMM 直到 dac-4 才出现可见下降，说明当前 CMM 等效模型中的 DAC 量化误差传播明显更弱
- 这进一步说明：当前 PyTorch CMM 等效仿真整体更“温和”，其 ADC/DAC 非理想性尚未复现 CrossSim/VTEAM 那种真实硬件路径中的严重累积误差

**核心结论**：

- **推荐 CMM DAC ≥ 6-bit**：6-bit 及以上基本无损；8-bit/10-bit/12-bit 无需刻意区分
- **dac-4 为明显退化起点**，但仍未达到 CrossSim 中 DAC 的灾难性退化程度
- 当前 CMM 中，DAC 也不是主要瓶颈；其影响弱于 `cell_bits`，并且显著弱于 CrossSim 中 DAC 的影响

### CMM Array Size 消融

**数据来源**：`checkpoints/cmm_array_size_conditions/metrics_pycoco.json`

| 条件 | array_size | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| **tile-64x64** | 64×64 | 0.6831 | 0.2479 | 0.2282 | 0.4959 | 0.8524 | 0.1676 |
| **tile-128x128** | 128×128 | 0.6818 | 0.2491 | 0.2293 | 0.4978 | 0.8517 | 0.1671 |
| **tile-256x256** | 256×256 | 0.6805 | 0.2470 | 0.2284 | 0.4967 | 0.8543 | 0.1672 |
| **tile-512x512** | 512×512 | 0.6812 | 0.2484 | 0.2293 | 0.4976 | 0.8544 | 0.1673 |

**分析**：

- **不同 tile size 下 caption 质量几乎不变**：四组条件的 BLEU-4 均在 0.2470 ~ 0.2491 之间，CIDEr 在 0.8517 ~ 0.8544 之间，与 baseline 的差异都非常小
- **不存在单调退化趋势**：64×64、128×128、256×256、512×512 各指标有轻微波动，但没有随着阵列增大或减小而持续恶化，说明当前 CMM 等效模型对 tile 切分并不敏感
- **128×128 与 512×512 略优但不具统计意义**：BLEU-4 分别为 0.2491、0.2484，较 baseline 略高，更像评估采样波动而非真实硬件收益
- **与 CrossSim array size 结果一致**：在当前无写噪声、无读噪声、连续 cell state 的理想 CMM 设置下，阵列规模本身不会成为 caption 质量瓶颈
- 这说明当前 PyTorch CMM 等效仿真里，tile 仅影响实现切分方式，不会像真实 crossbar 那样引入额外的阵列级非理想效应；后续仍需要在 CrossSim/真实器件模型下验证大阵列是否会带来 IR drop、失配累积等问题

**核心结论**：

- **当前 CMM 中 array size 不是精度瓶颈**：64×64 ~ 512×512 全范围基本无损
- **默认 128×128 配置可以继续沿用**，没有证据表明需要为了 caption 指标刻意缩小或放大 tile
- 若后续要研究阵列规模影响，重点应转向 **array size × 非理想因素联动**，而不是仅做理想 tile 切分

### CMM Write Noise 消融（多种子）

**数据来源**：`checkpoints/cmm_write_noise_conditions/metrics_mean_std.json` + `metrics_pycoco.json`（n=3 seeds: 1, 2, 3）

| noise_std | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6816 | 0.4998 | 0.3516 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| 0 | 0.6816 ± 0 | 0.4998 ± 0 | 0.3516 ± 0 | 0.2464 ± 0 | 0.2276 ± 0 | 0.4954 ± 0 | 0.8546 ± 0 | 0.1680 ± 0 |
| **1e-4** | 0.6823 ± 0.0004 | 0.5006 ± 0.0012 | 0.3526 ± 0.0016 | 0.2477 ± 0.0017 | 0.2278 ± 0.0005 | 0.4957 ± 0.0007 | 0.8548 ± 0.0035 | 0.1678 ± 0.0001 |
| **3e-4** | 0.6825 ± 0.0013 | 0.5012 ± 0.0013 | 0.3534 ± 0.0012 | 0.2487 ± 0.0012 | 0.2280 ± 0.0005 | 0.4961 ± 0.0004 | 0.8539 ± 0.0021 | 0.1673 ± 0.0002 |
| **1e-3** | 0.6816 ± 0.0050 | 0.5010 ± 0.0037 | 0.3528 ± 0.0025 | 0.2481 ± 0.0019 | 0.2278 ± 0.0009 | 0.4962 ± 0.0013 | 0.8540 ± 0.0004 | 0.1670 ± 0.0006 |
| 3e-3 | 0.6787 ± 0.0056 | 0.4995 ± 0.0051 | 0.3502 ± 0.0045 | 0.2444 ± 0.0035 | 0.2267 ± 0.0007 | 0.4956 ± 0.0006 | 0.8490 ± 0.0008 | 0.1671 ± 0.0008 |
| 1e-2 | 0.6739 ± 0.0133 | 0.4944 ± 0.0100 | 0.3467 ± 0.0088 | 0.2410 ± 0.0054 | 0.2245 ± 0.0019 | 0.4914 ± 0.0029 | 0.8395 ± 0.0070 | 0.1648 ± 0.0010 |

**分析**：

- **CMM 写入噪声影响远小于 CrossSim**：即使 std=1e-2，BLEU-4 仅从 0.2464 降至 0.2410（-2.2%），CIDEr 从 0.8546 降至 0.8395（-1.8%）；而 CrossSim write noise 在 it-6（1e-2）时 BLEU-4=0、CIDEr≈0（完全崩溃）
- **低噪声有微弱正效应**：1e-4 ~ 1e-3 区间 BLEU-4/CIDEr 均值略高于 baseline（BLEU-4 0.2477~0.2487 vs 0.2464），与 CrossSim read noise 的正则化效应类似
- **噪声 >= 1e-3 时种子间方差开始显著**：BLEU-1 std 从 0.0004 逐步增至 0.0133，CIDEr std 从 0 增至 0.0070
- **SPICE 在各噪声级别均保持极低波动**：std ≤ 0.0010，该指标对 CMM write noise 最不敏感
- **CMM 的 r-state clamp [0,1] 机制天然抑制噪声**：写入噪声作用在 r 上后再裁剪到物理合法范围，大幅削弱了噪声的逐层累积
- 物理直觉：CrossSim 的 programming_error 直接扰动电导值无上限裁剪，而 CMM 的 write_noise 作用在 [0,1] 归一化 r 上并 clamp，属于自带饱和的噪声模型

**核心结论**：CMM write noise 在现有参数下对模型质量影响极弱，即使 1e-2 也远未达到 CrossSim 同类噪声的破坏力。后续需在 CrossSim 接入阶段以真实器件模型验证。

### CMM Read Noise 消融（多种子）

**数据来源**：`checkpoints/cmm_read_noise_conditions/metrics_mean_std.json` + `metrics_pycoco.json`（n=3 seeds: 1, 2, 3）

| read_noise_std | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6816 | 0.4998 | 0.3516 | 0.2464 | 0.2276 | 0.4954 | 0.8546 | 0.1680 |
| 0 | 0.6816 ± 0 | 0.4998 ± 0 | 0.3516 ± 0 | 0.2464 ± 0 | 0.2276 ± 0 | 0.4954 ± 0 | 0.8546 ± 0 | 0.1680 ± 0 |
| **1e-4** | 0.6817 ± 0.0001 | 0.4999 ± 0.0000 | 0.3516 ± 0.0000 | 0.2464 ± 0.0000 | 0.2276 ± 0 | 0.4954 ± 0 | 0.8547 ± 0.0001 | 0.1681 ± 0.0001 |
| **3e-4** | 0.6817 ± 0.0000 | 0.5000 ± 0.0001 | 0.3520 ± 0.0004 | 0.2472 ± 0.0007 | 0.2277 ± 0.0002 | 0.4955 ± 0.0002 | 0.8556 ± 0.0005 | 0.1680 ± 0.0002 |
| **1e-3** | 0.6832 ± 0.0014 | 0.5019 ± 0.0019 | 0.3539 ± 0.0018 | 0.2490 ± 0.0018 | 0.2283 ± 0.0006 | 0.4964 ± 0.0010 | 0.8590 ± 0.0027 | 0.1684 ± 0.0005 |
| 3e-3 | 0.6833 ± 0.0016 | 0.5023 ± 0.0021 | 0.3544 ± 0.0022 | 0.2492 ± 0.0022 | 0.2285 ± 0.0006 | 0.4967 ± 0.0012 | 0.8588 ± 0.0049 | 0.1682 ± 0.0007 |
| 1e-2 | 0.6817 ± 0.0010 | 0.5004 ± 0.0002 | 0.3522 ± 0.0012 | 0.2479 ± 0.0022 | 0.2277 ± 0.0005 | 0.4959 ± 0.0003 | 0.8505 ± 0.0040 | 0.1673 ± 0.0004 |

**分析**：

- **CMM Read noise 几乎无影响**：所有 read noise 条件（含 1e-2）的指标与 baseline 在统计噪声范围内一致，与 CrossSim read noise 表现一致
- **1e-3 ~ 3e-3 区间有微弱正效应**：BLEU-4 0.2490~0.2492 vs baseline 0.2464（+1.1%），CIDEr 0.8590 vs 0.8546（+0.5%），与 CrossSim read noise 在 1e-3 处的正则化效应类似
- **1e-2 时正效应消退**：BLEU-4 0.2479 仍略高于 baseline，但低于 1e-3/3e-3 的峰值，CIDEr 回落到 0.8505（-0.5% vs baseline）
- **SPICE 极度稳定**：全量程 std ≤ 0.0007，mean 波动 < 0.001，该指标对 read noise 完全不敏感
- **与 CrossSim read noise 结果印证**：两种噪声模型下 read noise 均表现为无害甚至略有正向正则化，物理上因为 read noise 不改变存储状态，仅影响个别读取的瞬时值，在序列生成过程中会被平均化
- **种子间方差极小**：BLEU-4 最大 std 仅 0.0022（1e-2），远小于 write noise 同级别的 0.0054，说明 read noise 是多 token 平均自抵消的

**核心结论**：CMM read noise 完全无害，与 CrossSim read noise 结论一致。两种模型共同确认：read noise 不是 Transformer 图像描述任务的威胁，主要精度瓶颈来自写入端（write noise / cell_bits / DAC）。

---

### CMM 消融实验汇总

**已完成**：cell_bits 量化、ADC 分辨率、DAC 分辨率、array size、write noise（多种子）、read noise（多种子）六项 PyTorch CMM 消融。

| 消融项 | 变量范围 | 安全阈值 | 崩溃阈值 | 与 CrossSim 对比 |
| --- | --- | --- | --- | --- |
| cell_bits 量化 | 2-bit ~ continuous | ≥ 6-bit（CIDEr -2%） | 2-bit 崩溃 | 介于 ADC-6 与 ADC-8 之间，崩溃阈值更高 |
| ADC 分辨率 | 4-bit ~ ideal | ≥ 6-bit（adc-6 仅轻微退化） | 未见崩溃 | **显著弱于 CrossSim**：adc-4 仍可用 |
| DAC 分辨率 | 4-bit ~ ideal | ≥ 6-bit（dac-6 基本无损） | 未见崩溃 | **显著弱于 CrossSim**：dac-4 才明显下降 |
| Array size | 64×64 ~ 512×512 | 全范围无害 | 无 | 与 CrossSim 一致：理想条件下阵列切分不敏感 |
| Write noise | 0 ~ 1e-2（3 seeds） | ≤ 3e-3（BLEU-4 +0.9%） | 未见崩溃 | **远弱于 CrossSim**：CMM 1e-2 BLEU-4 -2.2% vs CrossSim 完全崩溃 |
| Read noise | 0 ~ 1e-2（3 seeds） | 全量程无害 | 无 | 与 CrossSim 一致：read noise 完全无害 |

**跨噪声类型对比（CMM 内部）**：

- **威胁排序**：cell_bits >> write noise > ADC ≈ DAC > read noise ≈ array size
- cell_bits 是 CMM 当前最严苛的精度瓶颈：4-bit 已造成明显退化，2-bit 直接崩溃
- ADC/DAC 在当前 CMM 等效模型中都较温和：6-bit 基本可用，4-bit 才进入明显退化区
- write noise 受 r-state clamp [0,1] 天然抑制，即使 1e-2 也远未崩溃
- read noise 与 array size 基本无害，与当前理想化 CMM 建模的鲁棒性一致

**总结**：

本阶段围绕 `CMMLinear` 论文式等效忆阻器模型，系统完成了 **cell_bits、ADC、DAC、array size、write noise、read noise** 六类消融实验，并在 COCO caption 任务上使用 BLEU、METEOR、ROUGE-L、CIDEr、SPICE 等指标进行了评估。整体结果表明，**当前 PyTorch CMM 等效模型对多数非理想因素表现出较强鲁棒性，真正显著影响性能的主要是 `cell_bits` 量化精度**。

从全部实验结果看，`cell_bits` 是当前最主要的精度瓶颈：`cell_bits >= 6` 时模型基本稳定，`cell_bits = 4` 时开始出现明显退化，而 `cell_bits = 2` 时已经无法支撑 caption 质量。相比之下，`write noise` 虽然会带来一定波动，但即使增大到 `1e-2` 也仅造成轻微下降，远弱于 CrossSim 中同级别写噪声的灾难性影响，这说明当前 CMM 中写噪声路径更“温和”，其一个重要原因是噪声作用在归一化 `r-state` 上并被 clamp 到 `[0,1]` 合法范围。

`read noise` 在全部测试范围内基本无害，甚至在 `1e-3 ~ 3e-3` 区间还出现轻微正向波动；这与 CrossSim 的结论一致，说明读取噪声不会改变存储状态，在序列生成任务中容易被多 token 平均化。与此同时，ADC/DAC 分辨率的影响也明显弱于预期：`8-bit` 及以上几乎无损，`6-bit` 仍基本可用，`4-bit` 才开始明显退化，但仍未达到 CrossSim 中相同条件下的严重崩溃程度。`array size` 则几乎不影响 caption 质量，`64×64 ~ 512×512` 之间没有观察到单调退化趋势，说明当前 CMM 等效模型对 tile 切分本身并不敏感。

综合来看，当前 CMM 消融实验说明：**PyTorch CMM 等效模型的主要威胁来源是状态量化误差，其次才是写入噪声，而 read noise、ADC/DAC 量化和 array size 的影响都相对较弱。** 这也意味着当前 CMM 建模整体偏理想化、偏温和，尚未复现 CrossSim/VTEAM 器件模型中更强烈的误差累积效应。因此，这一阶段实验的核心价值，一方面在于验证 CMM 映射与评估管线已经跑通，另一方面也为下一步的 **Module-wise sensitivity analysis** 与 **CrossSim 接入 CMM 对照实验** 提供了清晰的基线。

---

## 下一步计划

1. [x] **CrossSim DAC bit sweep** — 已完成（dac-12 ~ dac-4），DAC 影响远大于 ADC，推荐 ≥ 6-bit
1. [x] **CrossSim Read noise sweep** — 已完成（std 0 ~ 1e-2），read noise 几乎无影响
1. [x] **CrossSim Array size sweep** — 已完成（64×64 ~ 512×512），理想条件下无差异
1. [x] **CMM cell_bits 量化消融** — 已完成（2-bit ~ continuous），推荐 ≥ 6-bit
1. [x] **CMM Write noise 消融** — 已完成（0 ~ 1e-2, 3 seeds），影响远弱于 CrossSim
1. [x] **CMM Read noise 消融** — 已完成（0 ~ 1e-2, 3 seeds），完全无害
1. **Module-wise sensitivity analysis** — 分别对 attention 投影层（Q/K/V/O）和 FFN 层（FFN1/FFN2）施加噪声，定位最敏感模块
1. **CrossSim 接入 CMM** — 在 CrossSim 真实器件路径下复现 CMM 非理想效应（ADC/DAC、阵列尺寸、Ron/Roff 分布等）

---

## 后续计划

### 1. PyTorch CMM 等效仿真 ✅

验证 CMM 映射管线的正确性：

- [x] CMM 映射的正确性（`check_cmm_linear_equivalence.py`：理想条件下 `CMMLinear` ≡ `nn.Linear`，MAE ~ 1e-7）
- [x] `decoder_only` 替换的正确性（`evaluate_crosssim`：CMM 映射后 loss/acc 与 baseline 一致，MAE ~ 1e-6）
- [x] Caption 指标能否跑通（`compute_metrics_pycoco`：cell-continuous 与 baseline 完全一致）
- [x] cell_bits 量化消融（2-bit ~ continuous，推荐 ≥ 6-bit）
- [x] Write noise 消融（0 ~ 1e-2，3 seeds，影响远弱于 CrossSim）
- [x] Read noise 消融（0 ~ 1e-2，3 seeds，完全无害）

### 2. CrossSim 接入

在真实 crossbar 仿真路径下，系统性评估硬件非理想因素：

- [ ] ADC/DAC 分辨率对 CMM 映射结果的影响
- [ ] 阵列尺寸（tile shape）与非理想效应的交互
- [ ] Write noise + Read noise 联合注入
- [ ] 非理想因素（Ron/Roff 分布、stuck-at-fault、IR drop 等）建模
- [ ] CMM vs CrossSim 噪声模型差异的定量分析（r-state clamp 饱和效应 vs VTEAM 线性扰动）
