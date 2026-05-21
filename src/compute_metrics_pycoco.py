# 模型评测脚本：对比基线模型与忆阻器映射模型在 COCO 验证集上的描述生成指标
#
# 使用方式:
#   python -m src.compute_metrics_pycoco --baseline-checkpoint <基线检查点> --conditions-manifest <条件清单>
#
# 示例:
#   python -m src.compute_metrics_pycoco --baseline-checkpoint checkpoints/best.pt --conditions-manifest manifest.json --limit 500

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from .coco_preprocess.loader import DEFAULT_COCO_ROOT, default_image_transform
from .coco_preprocess.tokenizer import WordTokenizer
from .coco_preprocess.vocab import Vocabulary
from .map_memtorch import build_memristive_model, build_model_from_payload, load_checkpoint


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate caption models with pycocoevalcap")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True, help="基线模型检查点路径")
    parser.add_argument("--conditions-manifest", type=Path, required=True, help="忆阻器模型条件清单 JSON 文件")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT, help="COCO 数据集根目录")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/metrics_pycoco.json"), help="评测结果输出路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--max-len", type=int, default=30, help="生成描述的最大长度")
    parser.add_argument("--limit", type=int, default=1000, help="评测图片数量上限")
    return parser.parse_args()


def require_pycoco() -> Tuple[Any, Any]:
    """检查并导入 pycoco 评测工具（需在 mem 环境中安装）"""
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except Exception as e:
        raise RuntimeError(
            "pycocoevalcap/pycocotools is not available in current environment. "
            "Install both packages in mem env first."
        ) from e
    return COCO, COCOEvalCap


def build_vocab(payload: Dict[str, Any]) -> Vocabulary:
    """从检查点数据中恢复词表"""
    stoi = payload.get("vocab_stoi")
    if not stoi:
        raise ValueError("Checkpoint missing vocab_stoi")
    vocab = Vocabulary(tokenizer=WordTokenizer(), min_freq=1)
    vocab.stoi = stoi
    vocab.itos = {i: w for w, i in stoi.items()}  # 反向索引：id -> 词
    return vocab


def load_baseline_model(checkpoint: Path, device: torch.device):
    """加载基线模型（原始权重，未经忆阻器映射）"""
    payload, state = load_checkpoint(checkpoint, map_location="cpu")
    model = build_model_from_payload(payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def load_mem_model(mem_checkpoint: Path, baseline_model, device: torch.device):
    """加载忆阻器映射后的模型，使用基线模型结构 + 映射权重"""
    mem_payload = torch.load(mem_checkpoint, map_location="cpu")
    mem_args = mem_payload.get("memtorch_args", {})

    # 根据映射参数构建忆阻器模型
    mem_model = build_memristive_model(
        model=baseline_model,
        use_bindings=bool(mem_args.get("use_bindings", False)),
        tile_shape=tuple(mem_args.get("tile_shape", (128, 128))),
        max_input_voltage=float(mem_args.get("max_input_voltage", 0.3)),
        adc_resolution=int(mem_args.get("adc_resolution", 8)),
        ron=float(mem_args.get("r_on", 1e2)),
        roff=float(mem_args.get("r_off", 1e4)),
    )
    mem_model.load_state_dict(mem_payload["mem_model_state_dict"])
    mem_model.to(device)
    mem_model.eval()
    return mem_model, mem_payload


def generate_caption(model, image_tensor, vocab, device, max_len):
    """自回归生成图片描述文本"""
    bos_id = vocab.stoi["<bos>"]  # 起始标记
    eos_id = vocab.stoi["<eos>"]  # 结束标记
    tokens = model.generate(image=image_tensor.to(device), bos_id=bos_id, eos_id=eos_id, max_len=max_len)
    return vocab.decode(tokens.detach().cpu().tolist())  # token id -> 文本


def collect_image_ids(coco_annotation: Path, limit: int) -> List[int]:
    """从 COCO 标注文件中收集图片 ID 列表（受 limit 限制）"""
    with coco_annotation.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ids = [img["id"] for img in data["images"]]
    return ids[:limit] if limit > 0 else ids


def build_predictions_json(
    model,
    vocab: Vocabulary,
    coco_root: Path,
    image_ids: List[int],
    out_path: Path,
    device: torch.device,
    max_len: int,
):
    """对指定图片列表批量生成描述，保存为 COCO 格式的预测 JSON 文件"""
    ann_path = coco_root / "annotations" / "captions_val2014.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in data["images"]}  # 图片 id -> 文件名
    transform = default_image_transform(224)

    preds = []
    for image_id in tqdm(image_ids, desc=f"generate:{out_path.stem}", leave=False):
        file_name = id2file[image_id]
        img = Image.open(coco_root / "val2014" / file_name).convert("RGB")
        image_tensor = transform(img)
        caption = generate_caption(model, image_tensor, vocab, device, max_len=max_len)
        preds.append({"image_id": int(image_id), "caption": caption})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)


def evaluate_with_pycoco(COCO, COCOEvalCap, gt_ann: Path, pred_json: Path, image_ids: List[int]) -> Dict[str, float]:
    """使用 pycocoevalcap 计算 BLEU / METEOR / ROUGE_L 等指标"""
    coco = COCO(str(gt_ann))
    coco_res = coco.loadRes(str(pred_json))

    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = image_ids
    coco_eval.evaluate()

    metrics = {k: float(v) for k, v in coco_eval.eval.items()}
    return metrics


def main() -> None:
    """主流程：加载基线模型 -> 逐条件加载忆阻器模型 -> 生成预测 -> 计算指标并对比输出"""
    args = parse_args()
    COCO, COCOEvalCap = require_pycoco()

    device = torch.device(args.device)
    gt_ann = args.coco_root / "annotations" / "captions_val2014.json"
    image_ids = collect_image_ids(gt_ann, args.limit)

    # 加载基线模型与词表
    base_model, base_payload = load_baseline_model(args.baseline_checkpoint, device)
    vocab = build_vocab(base_payload)

    output_dir = args.output.parent / "predictions"
    rows: List[Dict[str, Any]] = []

    # 评测基线模型
    baseline_pred = output_dir / "baseline_predictions.json"
    build_predictions_json(base_model, vocab, args.coco_root, image_ids, baseline_pred, device, args.max_len)
    baseline_metrics = evaluate_with_pycoco(COCO, COCOEvalCap, gt_ann, baseline_pred, image_ids)
    rows.append({"model": "baseline", "checkpoint": str(args.baseline_checkpoint), "metrics": baseline_metrics})

    # 遍历条件清单，逐个评测忆阻器映射模型
    manifest = json.loads(args.conditions_manifest.read_text(encoding="utf-8"))
    for item in manifest:
        ckpt = Path(item["checkpoint"])
        cond = item["condition"]

        mem_model, _ = load_mem_model(ckpt, base_model, device)
        pred_file = output_dir / f"{cond}_predictions.json"
        build_predictions_json(mem_model, vocab, args.coco_root, image_ids, pred_file, device, args.max_len)
        metrics = evaluate_with_pycoco(COCO, COCOEvalCap, gt_ann, pred_file, image_ids)

        rows.append(
            {
                "model": cond,
                "noise_std": item.get("noise_std"),
                "checkpoint": str(ckpt),
                "metrics": metrics,
            }
        )

        del mem_model  # 及时释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保存汇总结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_images": len(image_ids),
                "image_id_range": [int(image_ids[0]), int(image_ids[-1])] if image_ids else [],
                "results": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 打印结果对比
    print(f"Saved metrics: {args.output}")
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['model']}: BLEU-1={m.get('Bleu_1', float('nan')):.4f}, "
            f"BLEU-4={m.get('Bleu_4', float('nan')):.4f}, "
            f"METEOR={m.get('METEOR', float('nan')):.4f}, ROUGE_L={m.get('ROUGE_L', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
