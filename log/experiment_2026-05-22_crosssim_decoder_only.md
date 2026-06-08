# CrossSim decoder_only 零损映射验证（MemTorch → CrossSim）

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-22（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | src/evaluate_crosssim.py |

## 实验目的

验证在放弃 MemTorch 后，CrossSim 的 `decoder_only` 映射是否能够在理想条件下近似无损复现数字基线；同时确认此前观察到的系统性偏差确实来自 MemTorch 的 `naive_map` / `naive_program` / `naive_scale` + VTEAM 组合，而不是 ADC resolution、tile shape、3D 输入适配或 `output_proj` 映射范围等因素。

## 运行命令

```bash
python -m src.evaluate_crosssim \
  --checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --crosssim-checkpoint caption_transformer_crosssim_decoder.pt \
  --device cuda \
  --max-batches 100
```

## 参数说明

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --checkpoint | checkpoints/caption_transformer_epoch_10.pt | 日志上下文 | 无（required） | Baseline training checkpoint |
| --crosssim-checkpoint | caption_transformer_crosssim_decoder.pt | 用户记录 | 无（required） | CrossSim checkpoint |
| --device | cuda | 用户记录 | `cuda` if available else `cpu` | 运行设备 |
| --batch-size | 16 | 默认值 | 16 | 验证 batch size |
| --num-workers | 0 | 默认值 | 0 | DataLoader worker 数 |
| --max-batches | 100 | 用户记录 | 100 | 最多评估 batch 数 |
| --subset-size | 0 | 默认值 | 0 | 0 表示使用完整验证集 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据根目录 |

## 实验结果

| 指标 | Baseline | CrossSim | 差值 |
|------|----------|----------|------|
| loss | 2.534256 | 2.534256 | 0.000000 |
| token_acc | 0.476609 | 0.476609 | 0.000000 |
| Logit MAE | — | 0.000001 | — |
| Logit Max Error | — | 0.000072 | — |
| Logit RMSE | — | 0.000001 | — |

## 备注

- 该实验对应一次关键路线决策：停止继续调 MemTorch，转向 CrossSim。
- 已排除的非根因因素包括：ADC resolution、tile shape、3D 输入适配、`output_proj` 映射范围。
- 结论：CrossSim `decoder_only` 映射几乎无损，误差已降至浮点舍入量级（Logit MAE ≈ `1e-6`）。
- MemTorch 失败的直接现象是权重级系统性偏差，Logit MAE 约为 `2.62`，且无法通过参数调优修复。
- 仍缺失的信息：`caption_transformer_crosssim_decoder.pt` 的完整相对路径，以及该次实验的精确执行时间。
