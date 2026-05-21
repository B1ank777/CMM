14. BLEU、METEOR、ROUGE 评价

第一版你可以用 nltk + rouge-score：

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np

def compute_metrics(references, hypotheses):
    """
    references: list[list[str]]
        每个样本可以有多个参考句子
    hypotheses: list[str]
        每个样本一个生成句子
    """

    smoothie = SmoothingFunction().method4

    refs_tokenized = [
        [tokenize(ref) for ref in refs]
        for refs in references
    ]
    hyps_tokenized = [
        tokenize(hyp)
        for hyp in hypotheses
    ]

    bleu1 = corpus_bleu(
        refs_tokenized,
        hyps_tokenized,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smoothie
    )
    bleu2 = corpus_bleu(
        refs_tokenized,
        hyps_tokenized,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothie
    )
    bleu3 = corpus_bleu(
        refs_tokenized,
        hyps_tokenized,
        weights=(1/3, 1/3, 1/3, 0),
        smoothing_function=smoothie
    )
    bleu4 = corpus_bleu(
        refs_tokenized,
        hyps_tokenized,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )

    meteor_scores = []
    for refs, hyp in zip(refs_tokenized, hyps_tokenized):
        meteor_scores.append(meteor_score(refs, hyp))

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []

    for refs, hyp in zip(references, hypotheses):
        best = 0.0
        for ref in refs:
            score = rouge.score(ref, hyp)["rougeL"].fmeasure
            best = max(best, score)
        rouge_l_scores.append(best)

    return {
        "BLEU1": bleu1,
        "BLEU2": bleu2,
        "BLEU3": bleu3,
        "BLEU4": bleu4,
        "METEOR": float(np.mean(meteor_scores)),
        "ROUGE_L": float(np.mean(rouge_l_scores))
    }

更严谨的论文实验建议使用 pycocoevalcap，因为 COCO captioning 标准评价通常会用它。