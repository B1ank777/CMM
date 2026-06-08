# 实验计划与进度

> 最后更新：2026-05-27

## 1. PyTorch CMM 等效仿真 ✅ 已完成

| 项目 | 状态 | 备注 |
|------|------|------|
| CMM 映射正确性 | ✅ | `check_cmm_linear_equivalence.py`：理想条件下 `CMMLinear` ≡ `nn.Linear`，MAE ~ 1e-7 |
| decoder_only 替换正确性 | ✅ | `evaluate_crosssim`：loss/acc 与 baseline 一致，MAE ~ 1e-6 |
| Caption 指标验证 | ✅ | `compute_metrics_pycoco`：cell-continuous 与 baseline 完全一致 |
| cell_bits 量化消融 | ✅ | 2-bit ~ continuous，推荐 ≥ 6-bit |
| Write noise 消融 | ✅ | 0 ~ 1e-2，3 seeds，影响远弱于 CrossSim |
| Read noise 消融 | ✅ | 0 ~ 1e-2，3 seeds，完全无害 |
| ADC 分辨率消融 | ✅ | 4 ~ 12-bit，推荐 ≥ 6-bit |
| DAC 分辨率消融 | ✅ | 4 ~ 12-bit，推荐 ≥ 6-bit |
| Array size 消融 | ✅ | 64×64 ~ 512×512，理想条件全范围无害 |

## 2. CrossSim 原生消融 ✅ 已完成

| 项目 | 状态 | 备注 |
|------|------|------|
| decoder_only 零损验证 | ✅ | Logit MAE ~ 1e-6 |
| Write noise 消融 | ✅ | it-10 ~ it-6，推荐 it-8 (1e-4) |
| Read noise 消融 | ✅ | 0 ~ 1e-2，完全无害 |
| ADC 分辨率消融 | ✅ | 4 ~ 12-bit，推荐 ≥ 10-bit |
| DAC 分辨率消融 | ✅ | 4 ~ 12-bit，DAC 是强敏感项 |
| Array size 消融 | ✅ | 64×64 ~ 512×512，理想条件全范围无害 |

## 3. CMM-CrossSim 两级映射 ⚙️ 进行中

| 项目 | 状态 | 备注 |
|------|------|------|
| 理想基线验证 | ✅ | ADC/DAC=0/0、无噪声，数值完全无损 |
| ADC 分辨率消融 | ✅ | 4 ~ 12-bit，推荐 ≥ 10-bit；CMM-CrossSim 比纯 CMM 对 ADC 更敏感 |
| DAC 分辨率消融 | ✅ | 4 ~ 12-bit，DAC 极强敏感，12-bit 仍有显著退化；无噪声对照确认退化主因是 DAC 量化本身 |
| Write noise 消融 | ✅ | 0 ~ 1e-2，3 seeds；≤ 3e-3 轻微波动，1e-2 明显退化且 seed 间方差增大 |
| cell_bits 量化消融 | ✅ | 2~8-bit + continuous；2-bit 功能崩溃，≥ 6-bit 推荐；CMM-CrossSim 比纯 CMM 对量化敏感得多 |
| Read noise 消融 | ✅ | 0 ~ 1e-2，3 seeds；全范围无害，种子间方差极小，与纯 CMM/CrossSim 结论一致 |
| Array size 消融 | ⬜ | 待做（checkpoint 已构建，待评测；重点：array size × 非理想因素联动） |

## 4. 模块级敏感性分析 ⬜ 待开始

分别对 attention 投影层（Q/K/V/O）和 FFN 层（FFN1/FFN2）施加噪声，定位最敏感模块。

| 项目 | 状态 | 备注 |
|------|------|------|
| Attention Q/K/V/O 逐模块 | ⬜ | |
| FFN1/FFN2 逐模块 | ⬜ | |
| output_proj (LM head) | ⬜ | |

## 5. 非理想因素深度建模 ⬜ 待开始

在真实 crossbar 仿真路径下，评估高级硬件非理想效应：

| 项目 | 状态 | 备注 |
|------|------|------|
| Ron/Roff 分布失配 | ⬜ | |
| Stuck-at-fault 器件 | ⬜ | |
| IR drop 效应 | ⬜ | |
| Write noise + Read noise 联合注入 | ⬜ | |
| CMM vs CrossSim 噪声模型定量对比 | ⬜ | r-state clamp 饱和效应 vs VTEAM 线性扰动 |

## 核心发现摘要

1. **CrossSim decoder_only 映射无损**：理想条件下 Logit MAE ~ 1e-6，验证了管线正确性。
2. **DAC 是主导瓶颈**：在 CrossSim 原生和 CMM-CrossSim 路径上，DAC 量化均远强于 ADC；CMM-CrossSim 中即便 dac-12 也已显著退化，且无噪声对照确认退化主因是 DAC 量化本身，而非 1e-4 读写噪声耦合。
3. **PyTorch CMM 过于温和**：纯 CMM 的 ADC/DAC 退化曲线显著平缓于 CrossSim 和 CMM-CrossSim，真实器件路径会放大 ADC/DAC 误差。
4. **Write noise 在 CMM 中被 `r`-state clamp 天然抑制**：1e-2 时 CMM 仅轻微退化，而 CrossSim 已完全崩溃。
5. **Read noise 在两种模型下均无害**：多 token 平均自抵消机制使其对序列生成几乎无影响。
6. **Array size 在理想条件下不是瓶颈**：64×64 ~ 512×512 全范围结果一致，真实影响可能在 × 非理想因素联动中显现。
