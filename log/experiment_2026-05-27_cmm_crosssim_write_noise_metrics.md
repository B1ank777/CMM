# CMM-on-CrossSim write noise 条件评测

| 属性 | 值 |
|------|-----|
| **日期** | 2026-05-27（具体时间未记录） |
| **状态** | 已完成 |
| **脚本** | `src/compute_metrics_pycoco.py` |

## 实验目的

对 `checkpoints/cmm_crosssim_write_noise_conditions/conditions_manifest.json` 中列出的 CMM-on-CrossSim 写噪声条件模型进行统一的 pycocoevalcap 评测，比较不同 `write_noise_std` 与不同 seed 下 caption 质量指标的变化，并判断该两阶段映射路径对写噪声的敏感区间。

## 运行命令

```bash
python -m src.compute_metrics_pycoco \
  --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt \
  --conditions-manifest checkpoints/cmm_crosssim_write_noise_conditions/conditions_manifest.json \
  --output checkpoints/cmm_crosssim_write_noise_conditions/metrics_pycoco.json \
  --limit 500
```

## 参数说明

| 参数 | 指定值 | 来源 | 默认值 | 说明 |
|------|--------|------|--------|------|
| --baseline-checkpoint | checkpoints/caption_transformer_epoch_10.pt | 用户指定 | 无（required） | 基线模型检查点路径 |
| --conditions-manifest | checkpoints/cmm_crosssim_write_noise_conditions/conditions_manifest.json | 用户指定 | 无（required） | 条件模型清单 JSON 文件 |
| --output | checkpoints/cmm_crosssim_write_noise_conditions/metrics_pycoco.json | 用户指定 | checkpoints/metrics_pycoco.json | 评测结果输出路径 |
| --limit | 500 | 用户指定 | 1000 | 评测图片数量上限 |
| --coco-root | data/coco | 默认值 | `data/coco` | COCO 数据集根目录 |
| --device | cuda / cpu（随环境自动选择） | 默认值 | `cuda` if available else `cpu` | 推理设备 |
| --max-len | 30 | 默认值 | 30 | 生成描述最大长度 |

## 实验结果

### pycocoevalcap 评测摘要（limit=500）

| 条件 | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|------|--------|--------|--------|---------|-------|-------|
| baseline | 0.6816 | 0.2464 | 0.2276 | 0.4954 | 0.8550 | 0.1680 |
| write-noise-0_seed-1 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8550 | 0.1680 |
| write-noise-0_seed-2 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8550 | 0.1680 |
| write-noise-0_seed-3 | 0.6818 | 0.2464 | 0.2276 | 0.4954 | 0.8550 | 0.1680 |
| write-noise-1e-04_seed-1 | 0.6842 | 0.2500 | 0.2288 | 0.4967 | 0.8590 | 0.1690 |
| write-noise-1e-04_seed-2 | 0.6815 | 0.2463 | 0.2276 | 0.4953 | 0.8540 | 0.1680 |
| write-noise-1e-04_seed-3 | 0.6829 | 0.2485 | 0.2282 | 0.4958 | 0.8560 | 0.1680 |
| write-noise-3e-04_seed-1 | 0.6817 | 0.2465 | 0.2280 | 0.4964 | 0.8560 | 0.1680 |
| write-noise-3e-04_seed-2 | 0.6800 | 0.2498 | 0.2274 | 0.4953 | 0.8570 | 0.1680 |
| write-noise-3e-04_seed-3 | 0.6848 | 0.2503 | 0.2282 | 0.4962 | 0.8470 | 0.1660 |
| write-noise-1e-03_seed-1 | 0.6809 | 0.2468 | 0.2278 | 0.4962 | 0.8510 | 0.1660 |
| write-noise-1e-03_seed-2 | 0.6787 | 0.2479 | 0.2263 | 0.4952 | 0.8540 | 0.1660 |
| write-noise-1e-03_seed-3 | 0.6808 | 0.2486 | 0.2298 | 0.4972 | 0.8580 | 0.1690 |
| write-noise-3e-03_seed-1 | 0.6784 | 0.2512 | 0.2279 | 0.4944 | 0.8470 | 0.1660 |
| write-noise-3e-03_seed-2 | 0.6773 | 0.2467 | 0.2286 | 0.4979 | 0.8580 | 0.1690 |
| write-noise-3e-03_seed-3 | 0.6716 | 0.2462 | 0.2262 | 0.4931 | 0.8460 | 0.1650 |
| write-noise-1e-02_seed-1 | 0.6526 | 0.2299 | 0.2239 | 0.4852 | 0.8000 | 0.1610 |
| write-noise-1e-02_seed-2 | 0.6804 | 0.2430 | 0.2240 | 0.4906 | 0.8380 | 0.1630 |
| write-noise-1e-02_seed-3 | 0.6604 | 0.2354 | 0.2243 | 0.4887 | 0.8250 | 0.1610 |

## 备注

- 本次运行只执行了指标计算脚本，没有重新生成条件模型；输入来自 `checkpoints/cmm_crosssim_write_noise_conditions/conditions_manifest.json`。
- 从结果看，`write_noise_std <= 3e-3` 时整体指标大多仍围绕 baseline 小幅波动，尚未出现稳定且大幅的系统性退化。
- `write_noise_std = 1e-2` 时退化开始明显，且不同 seed 间方差增大：`seed-1` 与 `seed-3` 的 BLEU-1 / BLEU-4 / CIDEr 均有明显下降，而 `seed-2` 相对较轻，说明高写噪声区已出现较强随机性。
- 零写噪声三次结果与 baseline 基本一致，说明评测流程本身稳定，两阶段 CMM-on-CrossSim 映射在无写噪声时未引入额外误差漂移。
- 日志中的 SPICE 计算伴随 Java 警告（`sun.misc.Unsafe`、`System::loadLibrary` restricted method），但评测正常完成并成功写出 `checkpoints/cmm_crosssim_write_noise_conditions/metrics_pycoco.json`。
- 其中部分条件在首次 SPICE 计算时触发 Stanford parsing pipeline 初始化，导致单次 SPICE 耗时略高（约 3.7s–5.5s）；这更像运行时初始化开销，而非模型本身差异。
