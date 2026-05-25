"""
按 write_noise_std 汇总 CMM 多 seed 评估结果，输出 mean ± std。

输入应为 src.compute_metrics_pycoco 生成的 metrics JSON；其中每个 CMM 条件行
需要包含 write_noise_std、seed 和 metrics 字段。
"""

# 运行示例：
#   python -m src.summarize_cmm_write_noise_metrics ^
#       --metrics checkpoints\cmm_write_noise_conditions\metrics_pycoco.json ^
#       --output checkpoints\cmm_write_noise_conditions\metrics_mean_std.json

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List


DEFAULT_METRICS = "Bleu_1,Bleu_2,Bleu_3,Bleu_4,METEOR,ROUGE_L"


def parse_args() -> argparse.Namespace:
    """解析 mean ± std 汇总参数。"""
    parser = argparse.ArgumentParser(description="Summarize CMM write_noise metrics by mean ± std")
    parser.add_argument("--metrics", type=Path, required=True, help="compute_metrics_pycoco 输出的 metrics JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/cmm_write_noise_conditions/metrics_mean_std.json"),
        help="mean ± std 汇总结果输出路径",
    )
    parser.add_argument("--metric-keys", type=str, default=DEFAULT_METRICS, help="逗号分隔的指标名列表")
    return parser.parse_args()


def parse_metric_keys(raw: str) -> List[str]:
    """解析需要汇总的指标名。"""
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("--metric-keys 至少需要一个指标名。")
    return keys


def finite_values(values: List[float]) -> List[float]:
    """过滤 NaN/inf，避免无效指标污染 mean ± std。"""
    return [value for value in values if math.isfinite(value)]


def summarize(metrics_payload: Dict[str, Any], metric_keys: List[str]) -> List[Dict[str, Any]]:
    """按 write_noise_std 聚合同一噪声强度下的多 seed 指标。"""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in metrics_payload.get("results", []):
        if row.get("model") == "baseline":
            continue
        if "write_noise_std" not in row or "seed" not in row:
            continue
        groups[str(row["write_noise_std"])].append(row)

    summaries: List[Dict[str, Any]] = []
    for write_noise_std in sorted(groups.keys(), key=lambda value: float(value)):
        rows = groups[write_noise_std]
        metric_summary: Dict[str, Dict[str, float]] = {}
        for key in metric_keys:
            values = finite_values([float(row.get("metrics", {}).get(key, float("nan"))) for row in rows])
            if not values:
                metric_summary[key] = {"mean": float("nan"), "std": float("nan")}
            elif len(values) == 1:
                metric_summary[key] = {"mean": values[0], "std": 0.0}
            else:
                metric_summary[key] = {"mean": mean(values), "std": stdev(values)}

        summaries.append(
            {
                "write_noise_std": write_noise_std,
                "n": len(rows),
                "seeds": sorted(str(row["seed"]) for row in rows),
                "metrics": metric_summary,
            }
        )
    return summaries


def main() -> None:
    """主入口：读取指标 JSON，输出 mean ± std 汇总。"""
    args = parse_args()
    metric_keys = parse_metric_keys(args.metric_keys)
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    summaries = summarize(payload, metric_keys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"source": str(args.metrics), "summary": summaries}, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {args.output}")
    for item in summaries:
        print(f"write_noise_std={item['write_noise_std']} (n={item['n']}, seeds={','.join(item['seeds'])})")
        for key in metric_keys:
            stats = item["metrics"][key]
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
