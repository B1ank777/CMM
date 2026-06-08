# CMM 论文式映射接入与初始验证

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-25（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/cmm.py`；`src/map_cmm.py`；`src/check_cmm_linear_equivalence.py`；`src/evaluate_crosssim.py` |

## 实验目的

在项目中接入论文式 CMM 等效映射，替代 CrossSim/VTEAM 的器件建模路径，先验证三件事：
1. `CMMLinear` 在理想条件下是否与 `nn.Linear` 数值等价；
2. `decoder_only` scope 是否正确，只映射 decoder 侧线性层而不误伤 encoder；
3. 将训练好的 CaptionTransformer 替换为 CMM 版本后，整模评估是否仍保持无损。

## 运行命令

### 1. CMM 单层等价性与 scope 验证

```bash
python -m src.check_cmm_linear_equivalence \
  --device cpu \
  --tile-rows 3 \
  --tile-cols 5
```

### 2. 将基线模型映射为 CMM decoder_only 模型

```bash
python -m src.map_cmm \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --output checkpoints/caption_transformer_cmm.pt \
  --device cuda \
  --scope decoder_only \
  --tile-rows 128 \
  --tile-cols 128 \
  --rmin 1000 \
  --rmax 100000 \
  --cell-bits 0 \
  --write-noise-std 0.0 \
  --read-noise-std 0.0 \
  --adc-resolution 0 \
  --dac-resolution 0
```

### 3. 对 CMM decoder_only 模型做整模评估

```bash
python -m src.evaluate_crosssim \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --crosssim-checkpoint checkpoints/caption_transformer_cmm.pt \
  --device cuda \
  --max-batches 100
```

## 参数说明

### 脚本 1：`src/check_cmm_linear_equivalence.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --device | cpu | 用户记录 | cpu | 检查设备 |
| --batch-size | 2 | 默认值 | 2 | 2D/3D 等价性检查的 batch size |
| --seq-len | 3 | 默认值 | 3 | 3D 输入序列长度 |
| --in-features | 8 | 默认值 | 8 | 单层线性层输入维度 |
| --out-features | 4 | 默认值 | 4 | 单层线性层输出维度 |
| --tile-rows | 3 | 用户记录 | 3 | CMM tile 行数 |
| --tile-cols | 5 | 用户记录 | 5 | CMM tile 列数 |
| --adc-resolution | 0 | 默认值 | 0 | ADC 分辨率 |
| --dac-resolution | 0 | 默认值 | 0 | DAC 分辨率 |
| --d-model | 16 | 默认值 | 16 | Dummy decoder 隐藏维度 |
| --num-heads | 4 | 默认值 | 4 | 注意力头数 |
| --ffn-dim | 32 | 默认值 | 32 | FFN 维度 |
| --num-layers | 2 | 默认值 | 2 | Dummy decoder 层数 |
| --vocab-size | 50 | 默认值 | 50 | Dummy output projection 维度 |
| --seed | 42 | 默认值 | 42 | 随机种子 |
| --tol | 1e-5 | 默认值 | 1e-5 | 最大误差容忍阈值 |

### 脚本 2：`src/map_cmm.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | 训练检查点路径 |
| --output | checkpoints/caption_transformer_cmm.pt | 用户记录 | checkpoints/caption_transformer_cmm.pt | CMM 映射后模型输出路径 |
| --device | cuda | 用户记录 | `cuda` if available else `cpu` | 映射设备 |
| --scope | decoder_only | 用户记录 | decoder_only | 映射范围 |
| --tile-rows | 128 | 用户记录 | 128 | CMM 阵列最大行数 |
| --tile-cols | 128 | 用户记录 | 128 | CMM 阵列最大列数 |
| --rmin | 1000 | 用户记录/默认值 | 1000.0 | Ron，器件最小电阻 |
| --rmax | 100000 | 用户记录/默认值 | 100000.0 | Roff，器件最大电阻 |
| --cell-bits | 0 | 用户记录/默认值 | 0 | 单元状态量化 bit 数；0 表示连续状态 |
| --write-noise-std | 0.0 | 用户记录/默认值 | 0.0 | 写入噪声强度 |
| --read-noise-std | 0.0 | 用户记录/默认值 | 0.0 | 读噪声强度 |
| --adc-resolution | 0 | 用户记录/默认值 | 0 | ADC 分辨率 |
| --dac-resolution | 0 | 用户记录/默认值 | 0 | DAC 分辨率 |

### 脚本 3：`src/evaluate_crosssim.py`

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户记录 | 无（required） | Baseline training checkpoint |
| --crosssim-checkpoint | checkpoints/caption_transformer_cmm.pt | 用户记录 | 无（required） | CMM checkpoint |
| --device | cuda | 用户记录 | `cuda` if available else `cpu` | 运行设备 |
| --batch-size | 16 | 默认值 | 16 | 验证 batch size |
| --num-workers | 0 | 默认值 | 0 | DataLoader worker 数 |
| --max-batches | 100 | 用户记录 | 100 | 最多评估 batch 数 |
| --subset-size | 0 | 默认值 | 0 | 0 表示使用完整验证集 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据根目录 |

## 实验结果

### 1. CMM 单层等价性验证

| 指标 | 值 |
|------|----|
| Device | cpu |
| Tile shape | (3, 5) |
| 2D MAE | 0.0000000894 |
| 2D Max error | 0.0000003576 |
| 3D MAE | 0.0000000484 |
| 3D Max error | 0.0000002384 |

### 2. decoder_only 映射范围验证

| 指标 | 值 |
|------|----|
| Expected decoder_only linear layers | 21 |
| Mapped CMMLinear layers | 21 |
| Encoder CMMLinear layers | 0 |

### 3. CMM decoder_only 整模评估（`evaluate_crosssim.py`, max-batches=100）

| 指标 | Baseline | CMM | 差值 |
|------|----------|-----|------|
| loss | 2.534256 | 2.534256 | 0.000000 |
| token_acc | 0.476609 | 0.476609 | 0.000000 |
| Logit MAE | — | 0.000001 | — |
| Logit Max Error | — | 0.000055 | — |
| Logit RMSE | — | 0.000001 | — |
| CMMLinear layers | — | 21 | — |
| Scope | — | decoder_only | — |
| Tile shape | — | (128, 128) | — |
| Rmin/Rmax | — | 1000.0 / 100000.0 | — |
| write/read noise std | — | 0.0 / 0.0 | — |

## 备注

- `CMMLinear` 的核心设计是内部维护 `r_pos` / `r_neg` 两个归一化忆阻状态，用论文式 `r` 状态替代 CrossSim 的 VTEAM 电导器件模型。[src/cmm.py](src/cmm.py)
- `map_cmm.py` 复用了 CrossSim 的 scope 机制，支持 `output_only`、`layers_only`、`decoder_only` 三种映射范围。[src/map_cmm.py:54-70](src/map_cmm.py#L54-L70)
- 理想条件下，`CMMLinear` 与 `nn.Linear` 的误差仅在 `1e-7` 量级，可视为浮点舍入误差。
- scope 检查结果表明：`decoder_only` 应映射的 21 个线性层全部被正确替换，encoder 中没有任何层被误映射。
- 整模评估结果进一步说明：CMM `decoder_only` 替换后，loss / token accuracy / logits 与数字基线保持无损一致。
- 到这一阶段，可以认为 PyTorch CMM 等效仿真管线已经完成了三层验证：单层等价性、scope 正确性、整模 decoder_only 正确性。
- 仍缺失的信息：新增 `src/cmm.py` 的精确创建时间；以及 `python -m src.evaluate_crosssim --checkpoint ...` 在原始实验中的完整命令行字符串。
