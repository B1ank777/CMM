# 模型评测脚本：对比基线模型与忆阻器映射模型在 COCO 验证集上的描述生成指标
#
# 使用方式:
#   python -m src.compute_metrics_pycoco --baseline-checkpoint <基线检查点> --conditions-manifest <条件清单>
#
# 示例:
#   python -m src.compute_metrics_pycoco --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt --conditions-manifest checkpoints/cmm_cell_bits_conditions/conditions_manifest.json --output checkpoints/cmm_cell_bits_conditions/metrics_pycoco.json --limit 500
#
#   # 大显存 GPU 上可提高 batch-size 和 num-workers，减少逐图推理造成的 GPU 空等
#   python -m src.compute_metrics_pycoco --baseline-checkpoint checkpoints/caption_transformer_epoch_10.pt --conditions-manifest checkpoints/cmm_crosssim_adc_conditions/conditions_manifest.json --batch-size 128 --num-workers 8

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .coco_preprocess.loader import DEFAULT_COCO_ROOT, default_image_transform
from .coco_preprocess.tokenizer import WordTokenizer
from .coco_preprocess.vocab import Vocabulary
from .map_crosssim import (
    build_model_from_crosssim_payload,
    build_model_from_payload,
    load_checkpoint,
    synchronize_crosssim_cores,
)
from .map_cmm import build_model_from_cmm_payload


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate caption models with pycocoevalcap")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True, help="基线模型检查点路径")
    parser.add_argument("--conditions-manifest", type=Path, required=True, help="CrossSim 条件模型清单 JSON 文件")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT, help="COCO 数据集根目录")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/metrics_pycoco.json"), help="评测结果输出路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--max-len", type=int, default=30, help="生成描述的最大长度")
    parser.add_argument("--limit", type=int, default=1000, help="评测图片数量上限")
    parser.add_argument("--batch-size", type=int, default=32, help="批量生成描述时的图片 batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 读图进程数")
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


def load_crosssim_model(crosssim_checkpoint: Path, baseline_model, device: torch.device):
    """加载 CrossSim 或 CMM 映射模型，使用 checkpoint 元数据重建结构。"""
    crosssim_payload = torch.load(crosssim_checkpoint, map_location="cpu")
    is_cmm_checkpoint = crosssim_payload.get("format") == "cmm_v1" or "cmm_model_state_dict" in crosssim_payload
    if is_cmm_checkpoint:
        cmm_args = crosssim_payload.get("cmm_args")
        if not cmm_args:
            raise ValueError("CMM checkpoint missing 'cmm_args'.")
        if "cmm_model_state_dict" not in crosssim_payload:
            raise ValueError("CMM checkpoint missing 'cmm_model_state_dict'.")
        cmm_model = build_model_from_cmm_payload(baseline_model, cmm_args, device)
        cmm_model.load_state_dict(crosssim_payload["cmm_model_state_dict"])
        cmm_model.to(device)
        cmm_model.eval()
        return cmm_model, crosssim_payload

    crosssim_args = crosssim_payload.get("crosssim_args")
    if not crosssim_args:
        raise ValueError("Checkpoint missing 'crosssim_args' or 'cmm_args'.")
    if "crosssim_model_state_dict" not in crosssim_payload:
        raise ValueError("Checkpoint missing 'crosssim_model_state_dict' or 'cmm_model_state_dict'.")

    crosssim_model = build_model_from_crosssim_payload(baseline_model, crosssim_args, device)
    crosssim_model.load_state_dict(crosssim_payload["crosssim_model_state_dict"])
    # load_state_dict 原地写入权重后，必须同步 CrossSim 内部 core 矩阵。
    synchronize_crosssim_cores(crosssim_model)
    crosssim_model.to(device)
    crosssim_model.eval()
    return crosssim_model, crosssim_payload


def set_eval_seed(seed_value: Any) -> None:
    """按 manifest 中的 seed 固定评估随机性，覆盖 CMM 读噪声等 forward 随机项。"""
    seed = int(seed_value)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_caption(model, image_tensor, vocab, device, max_len):
    """自回归生成图片描述文本"""
    bos_id = vocab.stoi["<bos>"]  # 起始标记
    eos_id = vocab.stoi["<eos>"]  # 结束标记
    tokens = model.generate(image=image_tensor.to(device), bos_id=bos_id, eos_id=eos_id, max_len=max_len)
    return vocab.decode(tokens.detach().cpu().tolist())  # token id -> 文本


class CocoEvalImageDataset(Dataset):
    """COCO 评测图片数据集，只返回生成描述所需的 image_id 与图像张量。"""

    def __init__(self, coco_root: Path, image_ids: List[int], id2file: Dict[int, str]) -> None:
        self.coco_root = coco_root
        self.image_ids = image_ids
        self.id2file = id2file
        self.transform = default_image_transform(224)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        image_id = int(self.image_ids[index])
        file_name = self.id2file[image_id]
        img = Image.open(self.coco_root / "val2014" / file_name).convert("RGB")
        return image_id, self.transform(img)


@torch.no_grad()
def generate_caption_batch(model, images: torch.Tensor, vocab: Vocabulary, device: torch.device, max_len: int) -> List[str]:
    """批量自回归生成描述，保持贪心解码逻辑与单图 generate 一致。"""
    model.eval()
    bos_id = vocab.stoi["<bos>"]
    eos_id = vocab.stoi["<eos>"]
    images = images.to(device, non_blocking=True)
    batch_size = images.size(0)

    tokens = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

    for _ in range(max_len - 1):
        logits = model(images, tokens)
        next_ids = logits[:, -1, :].argmax(dim=-1)
        # 已经生成 EOS 的样本继续填 EOS，避免不同长度样本破坏 batch 对齐。
        next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
        tokens = torch.cat([tokens, next_ids[:, None]], dim=1)
        finished |= next_ids.eq(eos_id)
        if bool(finished.all()):
            break

    return [vocab.decode(row.tolist()) for row in tokens.detach().cpu()]


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
    batch_size: int,
    num_workers: int,
):
    """对指定图片列表批量生成描述，保存为 COCO 格式的预测 JSON 文件"""
    ann_path = coco_root / "annotations" / "captions_val2014.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in data["images"]}  # 图片 id -> 文件名
    dataset = CocoEvalImageDataset(coco_root=coco_root, image_ids=image_ids, id2file=id2file)
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )

    preds = []
    for batch_ids, images in tqdm(loader, desc=f"generate:{out_path.stem}", leave=False):
        captions = generate_caption_batch(model, images, vocab, device, max_len=max_len)
        for image_id, caption in zip(batch_ids.tolist(), captions):
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
    """主流程：加载基线模型 -> 逐条件加载 CrossSim 模型 -> 生成预测 -> 计算指标。"""
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
    build_predictions_json(
        base_model,
        vocab,
        args.coco_root,
        image_ids,
        baseline_pred,
        device,
        args.max_len,
        args.batch_size,
        args.num_workers,
    )
    baseline_metrics = evaluate_with_pycoco(COCO, COCOEvalCap, gt_ann, baseline_pred, image_ids)
    rows.append({"model": "baseline", "checkpoint": str(args.baseline_checkpoint), "metrics": baseline_metrics})

    # 遍历条件清单，逐个评测 CrossSim 映射模型
    manifest = json.loads(args.conditions_manifest.read_text(encoding="utf-8"))
    for item in manifest:
        ckpt = Path(item["checkpoint"])
        cond = item["condition"]

        if "seed" in item:
            set_eval_seed(item["seed"])
        crosssim_model, _ = load_crosssim_model(ckpt, base_model, device)
        pred_file = output_dir / f"{cond}_predictions.json"
        build_predictions_json(
            crosssim_model,
            vocab,
            args.coco_root,
            image_ids,
            pred_file,
            device,
            args.max_len,
            args.batch_size,
            args.num_workers,
        )
        metrics = evaluate_with_pycoco(COCO, COCOEvalCap, gt_ann, pred_file, image_ids)

        rows.append(
            {
                "model": cond,
                "checkpoint": str(ckpt),
                # 保留 manifest 中除路径外的条件元数据，兼容 write/read noise 与 ADC/DAC sweep。
                **{key: value for key, value in item.items() if key not in {"condition", "checkpoint"}},
                "metrics": metrics,
            }
        )

        del crosssim_model  # 及时释放显存
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
