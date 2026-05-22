# 基于忆阻器模块交叉阵列的硬件友好型 Transformer 图像描述生成器

> A Hardware-Friendly Transformer Image Caption Generator Based on Crossbar Arrays of Memristor Modules

## 1. 研究背景与动机

忆阻器交叉阵列（Memristor Crossbar）能以 O(1) 的时间复杂度完成矩阵-向量乘法（VMM），是神经网络推理加速的理想硬件载体。然而，忆阻器的写入精度受限于编程脉冲的宽度和幅度，存在固有的写入误差。

LSTM-CMM 已在先前工作中被验证为一种可行的架构，但 Transformer 作为当前 NLP/CV 领域的主流架构，其大规模线性投影（Q/K/V/O、FFN）是否能被 CMM 有效支持？写入误差对 self-attention 和 FFN 两部分的影响孰重孰轻？这些问题尚未被系统性研究。

本项目旨在 **将 Transformer 解码器中的线性层映射到 CMM crossbar 上**，通过 CrossSim 进行误差注入仿真，评估不同写入精度（it-10 ~ it-6）、ADC/DAC 分辨率、读噪声和阵列规模下的模型性能退化，为硬件设计提供指导。

## 2. 系统架构

```text
┌─────────────────────────────────────────┐
│              Image Encoder              │
│            ResNet-50 (CPU/GPU)          │
└─────────────────┬───────────────────────┘
                  │ visual features (memory)
                  ▼
┌─────────────────────────────────────────┐
│          Transformer Decoder            │
│  ┌───────────────────────────────────┐  │
│  │ Token Embedding + Position Embed  │  │
│  └───────────────┬───────────────────┘  │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │     Input SRAM (digital)          │  │
│  └───────┬───────┬───────┬───────────┘  │
│          │ DAC   │ DAC   │ DAC           │
│          ▼       ▼       ▼              │
│  ┌──────┴───────┴───────┴──────────┐   │
│  │   CMM_Q  │  CMM_K  │  CMM_V    │   │  ← 关键写入误差注入点
│  └──────┬───────┬───────┬──────────┘   │
│         │ ADC   │ ADC   │ ADC           │
│         ▼       ▼       ▼              │
│  ┌──────────────────────────────────┐   │
│  │  Attention Score + Softmax      │   │  (数字/CMOS)
│  └───────────────┬──────────────────┘   │
│                  ▼                      │
│  ┌──────────────────────────────────┐   │
│  │        CMM_O (Out Proj)          │   │  ← 关键写入误差注入点
│  └───────────────┬──────────────────┘   │
│                  ▼                      │
│  ┌──────────────────────────────────┐   │
│  │   LayerNorm + Residual (digital) │   │
│  └───────────────┬──────────────────┘   │
│                  ▼                      │
│  ┌──────────────────────────────────┐   │
│  │   CMM_FFN1 → Activation →       │   │  ← 关键写入误差注入点
│  │         CMM_FFN2                  │   │
│  └───────────────┬──────────────────┘   │
│                  ▼                      │
│  ┌──────────────────────────────────┐   │
│  │   LayerNorm + Residual (digital) │   │
│  └───────────────┬──────────────────┘   │
│                  ▼                      │
│  ┌──────────────────────────────────┐   │
│  │    CMM_Vocab (LM Head)           │   │  ← 关键写入误差注入点
│  └───────────────┬──────────────────┘   │
│                  ▼                      │
│          Output Caption                 │
└─────────────────────────────────────────┘
```

**设计原则**：CMM 负责固定权重矩阵的 VMM 运算；动态 attention、softmax、LayerNorm 和 residual 由数字/CMOS 外围模块完成。

## 3. 研究阶段

### 阶段一：软件基线 ✅

在纯净软件环境中实现小型 image caption Transformer，作为无硬件误差的参考模型。

| 组件 | 配置 |
|------|------|
| Image Encoder | ResNet-50 |
| Decoder | 2-layer Transformer decoder |
| Embedding dim | 256 / 512 |
| Heads | 4 / 8 |
| FFN dim | 1024 / 2048 |
| Dataset | MS COCO Captions 2014 |
| Metrics | BLEU-1~4, METEOR, ROUGE |

### 阶段二：CMM 误差注入仿真（CrossSim）✅

使用 CrossSim 将 `nn.Linear` 层替换为 VTEAM 忆阻器交叉阵列等效模型，模拟 CMM 非理想特性：

```text
W_real → W_mapped → W_written = W_real + write_noise
```

对以下矩阵注入误差：

- **W_Q, W_K, W_V** — 自注意力/交叉注意力的 Q/K/V 投影
- **W_O** — 注意力输出投影
- **W_FFN1, W_FFN2** — 前馈网络两层
- **W_vocab** — 词表投影

**消融实验维度**：

| 实验维度 | 默认值 | 扫描范围 |
|----------|--------|----------|
| 写入噪声 (write_noise_std) | 0 | it-10 (1e-6) ~ it-6 (1e-2) |
| 读噪声 (read_noise_std) | 0 | 0 ~ 1e-2 |
| ADC 分辨率 | 10 bit | 4 ~ 12 bit |
| DAC 分辨率 | 12 bit | 4 ~ 12 bit |
| 阵列规模 (tile_rows × tile_cols) | 128×128 | 64×64 ~ 512×512 |

### 阶段三：硬件架构设计

将完整的 Transformer decoder block 映射为 CMM tile 结构，评估：

- 延迟（Latency）
- 能耗（Energy）
- 面积（Area）
- ADC/DAC 开销
- 写入误差敏感度

## 4. 实验对照组

| 组别 | 描述 |
|------|------|
| **Ref-Transformer** | 无硬件误差的软件模型 |
| **CrossSim-baseline** | CrossSim 理想映射（无噪声、无量化） |
| **CMM-it10** | 写入噪声 std = 1e-6 + 中等非理想 ADC/DAC |
| **CMM-it9** | 写入噪声 std = 1e-5 + 中等非理想 ADC/DAC |
| **CMM-it8** | 写入噪声 std = 1e-4 + 中等非理想 ADC/DAC（参考折中点） |
| **CMM-it7** | 写入噪声 std = 1e-3 + 中等非理想 ADC/DAC |
| **CMM-it6** | 写入噪声 std = 1e-2 + 中等非理想 ADC/DAC |
| **ADC/DAC 消融** | 不同 ADC (4~12 bit)、DAC (4~12 bit) 组合 |
| **读噪声消融** | 不同 read_noise_std (0 ~ 1e-2) |
| **阵列规模消融** | tile 64×64 / 128×128 / 256×256 / 512×512 |

## 5. 评价指标

| 指标类型 | 具体指标 |
|----------|----------|
| 文本质量 | BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE, METEOR |
| 硬件性能 | Latency, Energy, Area |
| 系统开销 | ADC/DAC overhead, Write-in error sensitivity |

## 6. 核心研究问题

1. 原论文 CMM 是否能支持 Transformer 中的大规模线性投影？
2. 写入误差对 attention 和 FFN 哪个部分影响更大？
3. it-8 是否仍然是 Transformer 架构下的最佳硬件折中点？
4. Linear attention 是否比 softmax attention 更适合 CMM 实现？
5. 相比 LSTM-CMM，Transformer-CMM 在 BLEU、能耗、延迟和面积上是否更优？

## 7. 项目结构

```text
CMM/
├── README.md
├── .gitignore
├── data/                                   # COCO 数据集（gitignored）
│   └── coco/
│       ├── train2014/
│       ├── val2014/
│       └── annotations/
├── src/
│   ├── train_captioner.py                  # 训练入口脚本
│   ├── preprocess.py                       # 预处理验证脚本
│   ├── map_crosssim.py                     # CrossSim 忆阻器交叉阵列映射
│   ├── evaluate_crosssim.py                # CrossSim 条件模型批量评估
│   ├── generate_caption.py                 # 单张图片描述生成
│   ├── compute_metrics_pycoco.py           # BLEU/METEOR/ROUGE 指标计算
│   ├── check_crosssim_linear_equivalence.py # CrossSim AnalogLinear 等价性验证
│   ├── test_crosssim_write_noise_conditions.py   # 写入噪声消融实验
│   ├── test_crosssim_read_noise_conditions.py    # 读噪声消融实验
│   ├── test_crosssim_adc_conditions.py           # ADC 分辨率消融实验
│   ├── test_crosssim_dac_conditions.py           # DAC 分辨率消融实验
│   ├── test_crosssim_array_size_conditions.py    # 阵列规模消融实验
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                      # ResNet-50 图像编码器
│   │   ├── decoder.py                      # 多头注意力、解码器层
│   │   └── caption_transformer.py          # 完整 CaptionTransformer 模型
│   └── coco_preprocess/
│       ├── __init__.py
│       ├── coco_io.py                      # COCO JSON 标注加载
│       ├── tokenizer.py                    # 分词器
│       ├── vocab.py                        # 词表构建与编解码
│       ├── dataset.py                      # PyTorch Dataset 与 collate_fn
│       └── loader.py                       # DataLoader 构建与图像变换
└── checkpoints/                            # 训练检查点（运行时生成）
```

## 8. 环境配置

### 依赖安装

```bash
conda create -n mem python=3.10
conda activate mem

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tqdm pycocoevalcap nltk crosssim
```

### 数据集准备

从 [COCO 官网](https://cocodataset.org/) 下载：

- `train2014.zip` (13GB) — 训练集图像
- `val2014.zip` (6GB) — 验证集图像
- `annotations_trainval2014.zip` (241MB) — 标注文件

解压到 `data/coco/` 目录：

```text
data/coco/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

## 9. 使用指南

### 训练基线模型

```bash
# 默认配置（8GB 显存友好）
python -m src.train_captioner

# 自定义配置
python -m src.train_captioner \
    --d-model 512 \
    --num-heads 8 \
    --ffn-dim 2048 \
    --num-layers 2 \
    --epochs 10 \
    --batch-size 24 \
    --lr 1e-4 \
    --coco-root data/coco \
    --save-dir checkpoints

# 微调 CNN 编码器
python -m src.train_captioner --train-cnn --lr 1e-5

# 梯度累积（模拟更大 batch size）
python -m src.train_captioner --batch-size 16 --accum-steps 2
```

### 训练参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--coco-root` | Path | `data/coco` | COCO 数据集根目录 |
| `--save-dir` | Path | `checkpoints` | 检查点保存目录 |
| `--d-model` | int | 256 | 隐藏维度 |
| `--num-heads` | int | 4 | 注意力头数 |
| `--ffn-dim` | int | 1024 | FFN 中间维度 |
| `--num-layers` | int | 2 | 解码器层数 |
| `--max-len` | int | 30 | 最大描述长度 |
| `--min-freq` | int | 5 | 词表最低词频 |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch-size` | int | 24 | 批次大小 |
| `--num-workers` | int | 4 | 数据加载线程数 |
| `--lr` | float | 1e-4 | 学习率 |
| `--weight-decay` | float | 1e-4 | 权重衰减 |
| `--grad-clip` | float | 1.0 | 梯度裁剪阈值 |
| `--accum-steps` | int | 1 | 梯度累积步数 |
| `--no-amp` | flag | False | 禁用混合精度 |
| `--train-cnn` | flag | False | 解冻 CNN 编码器微调 |
| `--seed` | int | 42 | 随机种子 |

### 验证预处理管线

```bash
python -m src.preprocess
```

### CrossSim 映射

```bash
# 默认中等非理想映射（ADC=10 bit, DAC=12 bit），仅映射 decoder 线性层
python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt

# 理想 CrossSim 基线（关闭 ADC/DAC 量化）
python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --adc-resolution 0 --dac-resolution 0

# 自定义 tile 尺寸与 ADC/DAC 分辨率
python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --tile-rows 64 --tile-cols 64 --adc-resolution 8 --dac-resolution 8

# GPU 加速映射
python -m src.map_crosssim --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --device cuda --use-gpu
```

### CrossSim 映射参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | Path | 必填 | 训练检查点路径 |
| `--output` | Path | `checkpoints/caption_transformer_crosssim.pt` | 输出路径 |
| `--device` | str | `cuda` (可用时) / `cpu` | 设备 |
| `--use-gpu` | flag | False | 显式启用 CrossSim GPU 后端 |
| `--scope` | str | `decoder_only` | 映射范围：`output_only` / `layers_only` / `decoder_only` |
| `--tile-rows` | int | 128 | 交叉阵列最大行数 |
| `--tile-cols` | int | 128 | 交叉阵列最大列数 |
| `--adc-resolution` | int | 10 | ADC 分辨率；0 表示理想 ADC |
| `--dac-resolution` | int | 12 | DAC 分辨率；0 表示理想 DAC |
| `--bias-rows` | int | 0 | bias 额外行数；0 表示数字 bias |
| `--rmin` | float | 1e3 | 器件最小电阻 |
| `--rmax` | float | 1e5 | 器件最大电阻 |
| `--cell-bits` | int | 0 | 单元量化 bit；0 表示连续电导 |
| `--read-noise-std` | float | 0.0 | 读噪声强度；0 表示关闭 |
| `--programming-error-std` | float | 0.0 | 写入误差强度；0 表示关闭 |

### 消融实验

```bash
# 写入噪声消融（it-10 ~ it-6），中等非理想 ADC/DAC
python -m src.test_crosssim_write_noise_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/crosssim_write_noise_conditions \
    --save-baseline-crosssim

# 读噪声消融
python -m src.test_crosssim_read_noise_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/crosssim_read_noise_conditions

# ADC 分辨率消融
python -m src.test_crosssim_adc_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/crosssim_adc_conditions \
    --save-baseline-crosssim

# DAC 分辨率消融
python -m src.test_crosssim_dac_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/crosssim_dac_conditions \
    --save-baseline-crosssim

# 阵列规模消融
python -m src.test_crosssim_array_size_conditions \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --output-dir checkpoints/crosssim_array_size_conditions
```

### 模型评估

```bash
# 评估单个 CrossSim 条件模型
python -m src.evaluate_crosssim \
    --checkpoint checkpoints/crosssim_write_noise_conditions/caption_transformer_it-8_write_noise_crosssim.pt \
    --output-dir eval_results

# 批量评估所有条件（需先生成 manifest JSON）
python -m src.compute_metrics_pycoco \
    --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --conditions-manifest checkpoints/crosssim_write_noise_conditions/conditions_manifest.json \
    --limit 500
```

### 单图描述生成

```bash
python -m src.generate_caption \
    --checkpoint checkpoints/caption_transformer_epoch_10.pt \
    --image path/to/image.jpg

# 使用 CrossSim 映射模型
python -m src.generate_caption \
    --checkpoint checkpoints/caption_transformer_crosssim.pt \
    --image path/to/image.jpg
```

### CrossSim 等价性验证

```bash
python -m src.check_crosssim_linear_equivalence
```

## 10. 开发路线图

- [x] ResNet 图像编码器
- [x] Transformer 解码器（多头注意力 + 解码器层）
- [x] CaptionTransformer 端到端模型
- [x] COCO 数据预处理管线
- [x] 训练脚本（AMP + 梯度累积 + 检查点）
- [x] CrossSim 忆阻器交叉阵列映射（替换 MemTorch）
- [x] 评估脚本（BLEU / METEOR / ROUGE）
- [x] 单图描述生成脚本
- [x] 写入噪声消融实验（it-10 ~ it-6）
- [x] 读噪声消融实验
- [x] ADC 分辨率消融实验
- [x] DAC 分辨率消融实验
- [x] 阵列规模消融实验
- [ ] VGG-16 编码器支持
- [ ] Mem32 对照实验
- [ ] 硬件性能建模（延迟/能耗/面积）
- [ ] CMM tile 结构映射设计（阶段三）
