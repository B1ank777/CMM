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
- [x] MemTorch 映射脚本（map_memtorch.py）
- [x] 批量写入噪声条件构建脚本（test_memtorch_conditions.py）
- [x] MemTorch decoder-only 映射误差消融（ADC/tile/3D-input 均排除，误差源于权重本身）
- [x] 更换 CrossSim 替代 MemTorch
- [ ] VGG-16 编码器支持
- [ ] it-10 ~ it-6 写入误差实验运行
- [ ] Mem32 对照实验
- [ ] 硬件性能建模（延迟/能耗/面积）

### 待提交更改

- `src/compute_metrics_pycoco.py` — 评估指标计算脚本
- `src/map_memtorch.py` — MemTorch 映射修改
- `src/test_memtorch_conditions.py` — 条件测试修改
- `only_decoder.md` — 仅 patch decoder 的设计方案

### 设计决策

- 采用仅 patch decoder 的方案，避免对 CNN encoder 进行不必要的 MemTorch 映射（见 `only_decoder.md`）

---

## 实验记录

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
| BLEU-1 | |
| BLEU-2 | |
| BLEU-3 | |
| BLEU-4 | |
| METEOR | |
| ROUGE | |

### CMM 写入噪声实验

| 条件 | 噪声标准差 | BLEU-1 | BLEU-4 | METEOR | ROUGE |
|------|-----------|--------|--------|--------|-------|
| Ref-Transformer | 0 | | | | |
| it-10 | 1e-6 | | | | |
| it-9 | 1e-5 | | | | |
| it-8 | 1e-4 | | | | |
| it-7 | 1e-3 | | | | |
| it-6 | 1e-2 | | | | |

---

## 问题与发现

### 2026-05-21 — MemTorch Decoder-Only 映射后模型输出严重退化

**测试方式**：使用 `generate_caption` 分别加载基线模型和 MemTorch 映射后的 decoder-only 模型，对 3 张图片生成描述。

**结果**：
- `caption_transformer_epoch_10.pt`（基线模型）：3 张图片均得到有效描述
- `caption_transformer_memtorch_decoder_only.pt`（MemTorch 映射后）：3 张图片输出均为 `a <unk> <unk> ... <unk>`（全 `<unk>` 序列）

**初步判断**：当前 MemTorch 映射条件下，模型权重已严重失真，无法正常解码。需进一步量化退化程度。

**下一步**：运行 `evaluate_memtorch` / `compute_metrics_pycoco` 量化与 baseline 的 BLEU/METEOR/ROUGE 差异。

### 2026-05-21 — Decoder-Only MemTorch 评估量化结果

**测试方式**：`python -m src.evaluate_memtorch --checkpoint checkpoints/caption_transformer_epoch_10.pt --mem-checkpoint checkpoints/caption_transformer_memtorch_decoder_only.pt --num-workers 2`

```text
Device: cuda
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
MemTorch  loss: 5.823996, token_acc: 0.142226
Delta loss (mem - base): 3.289740
Delta acc  (mem - base): -0.334383
Logit MAE: 2.620198
```

**分析**：
- MemTorch 映射后 loss 从 2.53 飙升至 5.82（+130%），token 准确率从 47.7% 暴跌至 14.2%（-33.4 个百分点）
- Logit MAE 高达 2.62，说明权重映射导致输出 logits 严重偏移
- 与之前 `generate_caption` 的定性结果一致：映射后模型已实质性失效

**下一步**：暂不直接跑全量 BLEU/METEOR/ROUGE 指标，先做消融实验缩小误差来源。

---

## 下一步计划：MemTorch 映射误差消融

当前最优先的任务是定位 decoder-only mapping 中误差的主要来源，而非直接跑写入噪声实验。

### 消融路线图（按顺序执行）

1. [x] **仅 patch `output_proj`**：将 MemTorch 映射限定在最后的 LM head，其余 decoder 层保持原始 `nn.Linear`，排除层内投影的干扰
2. [x] **仅 patch layers（Q/K/V/O + FFN1/FFN2）**：保持 `output_proj` 为原始线性层，确认 decoder 内部 attention/FFN 的映射贡献
3. [x] **patch layers + output_proj（当前方案对照）**：复现已有结果作为 baseline 对照
4. [ ] **调整 `ADC_resolution`**：试 10-bit、12-bit，观察 ADC 量化精度对退化程度的影响
5. [ ] **调整 `tile_shape`**：增大 tile 尺寸减少切 tile 带来的边界误差，观察 tile 相关退化贡献
6. [ ] **对 it-10 / it-9 / it-8 分别做 1-5 的对比**：确认不同写入噪声水平下各消融项的变化趋势，定位最敏感组件

### 判断标准

- 每个消融项跑完后对比 `evaluate_memtorch` 输出的 **loss / token_acc / Logit MAE**，不再直接跑 pycocoevalcap 全量指标
- 找到使 Logit MAE 显著下降的变量即为主要误差来源

### 消融结果

#### 1. 仅 patch `output_proj`（`caption_transformer_mem_output_only.pt`）

```text
Device: cuda
Mem scope: output_only
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
MemTorch  loss: 2.534256, token_acc: 0.476609
Delta loss (mem - base): 0.000000
Delta acc  (mem - base): 0.000000
Logit MAE: 0.000000
```

**结论**：`output_proj` 单独映射对模型输出无任何影响（Logit MAE = 0）。误差来源在 decoder 层内的 Q/K/V/O / FFN1/FFN2 映射，与 `output_proj` 无关。

#### 2. 仅 patch layers（`caption_transformer_mem_layers_only.pt`）

```text
Device: cuda
Mem scope: layers_only
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
MemTorch  loss: 5.823996, token_acc: 0.142226
Delta loss (mem - base): 3.289740
Delta acc  (mem - base): -0.334383
Logit MAE: 2.620198
```

Patched 层明细：2 层 × 10 个 Linear（Q/K/V/O × 4 + FFN1/FFN2 × 2）× 2 = 20 个，均为 `(256→256)` 或 `(256↔1024)`。

**结论**：仅 patch layers 的结果与 layers + output_proj 完全相同（loss / acc / MAE 三位小数点一致）。确认 **误差 100% 来自 decoder 层内映射**，`output_proj` 不贡献任何退化。下一步重点排查 ADC resolution 和 tile_shape 两个参数。

#### 3. ADC_resolution = 10（`caption_transformer_mem_decoder_only_adc10.pt`）

```text
Device: cuda
Mem scope: decoder_only
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
MemTorch  loss: 5.824000, token_acc: 0.142281
Delta loss (mem - base): 3.289744
Delta acc  (mem - base): -0.334328
Logit MAE: 2.620211
Logit Max Error: 25.645622
Logit RMSE: 3.416300
```

ADC=8 对照（之前 decoder_only 结果）：loss=5.823996, acc=0.142226, MAE=2.620198。

**结论**：ADC 从 8-bit 提升到 10-bit **无任何改善**（MAE 差值 < 0.00002，属浮点舍入误差）。ADC 量化精度不是误差来源。继续试 adc=12 确认趋势后进入 tile_shape 排查。

#### 4. ADC_resolution = 12（`caption_transformer_mem_decoder_adc12.pt`）

```text
Device: cuda
Mem scope: decoder_only
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
MemTorch  loss: 5.823999, token_acc: 0.142281
Delta loss (mem - base): 3.289743
Delta acc  (mem - base): -0.334328
Logit MAE: 2.620212
Logit Max Error: 25.645666
Logit RMSE: 3.416301
```

| ADC | Loss     | Token Acc | Logit MAE | Logit Max Error | Logit RMSE |
|-----|----------|-----------|-----------|-----------------|------------|
| 8   | 5.823996 | 0.142226  | 2.620198  | —               | —          |
| 10  | 5.824000 | 0.142281  | 2.620211  | 25.645622       | 3.416300   |
| 12  | 5.823999 | 0.142281  | 2.620212  | 25.645666       | 3.416301   |

**结论**：ADC 8/10/12 三位小数点完全一致。**ADC 分辨率与当前误差无关**，可彻底排除。下一步排查 tile_shape。

#### 5. tile_shape = 256×256（`caption_transformer_mem_decoder_tile256.pt`）

```text
Device: cuda
Mem scope: decoder_only
Batches evaluated: 100
Baseline  loss: 2.534256, token_acc: 0.476609
MemTorch  loss: 5.823989, token_acc: 0.142281
Delta loss (mem - base): 3.289734
Delta acc  (mem - base): -0.334328
Logit MAE: 2.620205
Logit Max Error: 25.645140
Logit RMSE: 3.416297
```

| tile_shape | Loss     | Token Acc | Logit MAE | Logit Max Error | Logit RMSE |
|------------|----------|-----------|-----------|-----------------|------------|
| 128×128    | 5.823996 | 0.142226  | 2.620198  | —               | —          |
| 256×256    | 5.823989 | 0.142281  | 2.620205  | 25.645140       | 3.416297   |

**结论**：tile_shape 从 128→256 无任何改善。tile 分块量化也不是误差来源。至此，ADC resolution 和 tile_shape 两个常规参数均已排除。

**初步判断**：误差可能来自 MemTorch 底层 VTEAM 模型的 device-level 行为（naive_map / naive_program / naive_scale 的组合映射策略、ron/roff 器件参数、或 transistor 接入引入的非线性）。下一步可尝试：

- 增大 ron/roff 比值（当前 1e2/1e4，可试 1e3/1e6 或理想化参数）
- 关闭 transistor（`--transistor` flag 需加入 map_memtorch.py）
- 直接对比映射前后的逐层权重差异（weight MAE per layer）定位最敏感的层

### 2026-05-21 — 3D 输入显式展平消融

**测试方式**：修改 `MultiHeadAttention`，将 Q/K/V/O 投影改为显式 `[B,T,D] → [B*T,D] → Linear → [B,T,D_out]`；修改 `DecoderLayer.forward` 显式调用 `self.ffn[0]` / `self.ffn[1]` 等子层，确保进入 MemTorch Linear 的输入始终是 2D，排除 `patch_memtorch_linear_input_shapes` 中 reshape 工作区的潜在问题。

**结果**（`caption_transformer_mem_decoder_tile256.pt`，tile=256×256, adc=8）：

```text
MemTorch loss: 5.823989, token_acc: 0.142281
Logit MAE: 2.620205
```

与未做显式 2D 展平的 decoder_only 结果（MAE=2.620198）完全一致。

**结论**：`patch_memtorch_linear_input_shapes` 的 reshape 工作区没有问题。3D→2D→3D 的维度变换即使在代码层面显式写死，误差不变。问题不在输入形状适配，而在 MemTorch 映射后**权重本身的数值偏差**。

---

## 2026-05-22

### 决策：更换模拟库 MemTorch → CrossSim

**背景**：经过一整天的消融实验，排除了 ADC resolution（8/10/12 无差异）、tile_shape（128/256 无差异）、3D 输入适配（显式 2D 展平无差异）、output_proj 映射范围。所有表面参数调优均无法缩小 ~2.62 的 Logit MAE，误差定位为 MemTorch naive_map/naive_program/naive_scale + VTEAM 器件模型组合导致的**权重级系统性偏差**。

**决定**：放弃 MemTorch，改用 **CrossSim**（Sandia 国家实验室的 crossbar 仿真框架）作为忆阻器交叉阵列建模引擎。

**理由**：
- MemTorch 的 naive_* 映射策略对 Transformer 权重矩阵的数值保真度不足，底层的 VTEAM 模型 + transistor 引入的非线性难以调参纠正
- CrossSim 提供了更成熟的 ADC/DAC 建模、更灵活的分块策略、以及可配置的器件噪声模型
- CrossSim 支持 Keras/TensorFlow 接口的原生权重映射（`map()` API），同时底层与 PyTorch 权重数组兼容，可以复用现有 PyTorch 模型结构

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

| 指标 | MemTorch | CrossSim |
| --- | --- | --- |
| Loss | 5.823996 | **2.534256** |
| Token Acc | 14.2% | **47.7%** |
| Logit MAE | 2.62 | **1e-6** |
| Logit Max Error | 25.6 | **7.2e-5** |

**结论**：CrossSim 映射几乎无损（Logit MAE ~ 1e-6，仅为浮点舍入误差级别）。MemTorch 的问题彻底坐实为其底层 naive 映射策略 + VTEAM 器件模型导致。CrossSim 可以作为可靠的忆阻器仿真后端，可以继续写入噪声实验。

**迁移计划**：
- [x] 安装 CrossSim 环境并验证 GPU 兼容性
- [x] 实现 CrossSim 的 CrossSimLinear 层，替换 nn.Linear 作为映射目标
- [x] 重写与 map_memtorch.py 同级的 map_crosssim.py，支持 scope 选择
- [x] 用相同基线 checkpoint 跑 decoder_only 映射 → evaluate 对比 Logit MAE
- [x] 若 Logit MAE 显著低于 2.62，继续之前未完成的写入噪声实验

### CrossSim 写入噪声实验：it-6

**首个写入噪声条件**（`crosssim_conditions/caption_transformer_it-6_crosssim.pt`，噪声 std=1e-2）：

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
| MemTorch 无噪声 | 5.823996 | 14.2% | 2.620 | 25.65 | 3.416 |

**分析**：

- it-6 写入噪声下 CrossSim 仍有 22.5% token 准确率，优于 MemTorch **无噪声** 基线（14.2%）
- CrossSim it-6 的 MAE（1.991）也比 MemTorch 无噪声（2.620）低 24%，再次确认 MemTorch 底层映射偏差本身就已超过 it-6 噪声的影响量级
- it-6 噪声导致的 MAE 从 ~0 升至 1.99，说明 CrossSim 可以捕捉到写入噪声的退化效应，且数值在合理范围内

**下一步**：继续跑 it-10 ~ it-6 全系列，建立 CrossSim 的写入噪声退化曲线。

---

## 后续计划（CrossSim 迁移完成后）

- [ ] 运行 it-10 ~ it-6 写入噪声实验并记录指标
- [ ] 分析 attention 与 FFN 对不同噪声水平的敏感度
- [ ] 确定 it-8 是否为 Transformer 架构下的最佳折中点
