from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .tokenizer import BaseTokenizer


# 特殊标记：填充、句首、句尾、未知词
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Vocabulary:
    """词汇表：根据语料构建词表，提供文本↔索引的双向转换"""

    def __init__(self, tokenizer: BaseTokenizer, min_freq: int = 5):
        self.tokenizer = tokenizer
        self.min_freq = min_freq  # 最低词频阈值，低于此频率的词将被过滤
        self.stoi: Dict[str, int] = {}  # 词→索引
        self.itos: Dict[int, str] = {}  # 索引→词

    @property
    def pad_id(self) -> int:
        """返回 <pad> 对应的索引，用于填充"""
        return self.stoi["<pad>"]

    def build(self, captions: Iterable[str]) -> None:
        """基于所有标题文本构建词汇表"""
        counter = Counter()
        for cap in captions:
            counter.update(self.tokenizer.tokenize(cap))

        # 特殊标记始终保留，再按 min_freq 过滤低频词
        words = SPECIAL_TOKENS[:]
        words.extend([w for w, c in counter.items() if c >= self.min_freq])

        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text: str, max_len: int = 30) -> List[int]:
        """将文本编码为索引序列，自动添加 <bos>/<eos>，超出长度则截断"""
        if not self.stoi:
            raise ValueError("Vocabulary is empty. Call build() first.")
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to fit <bos> and <eos>.")

        # 预留 <bos> 和 <eos> 两个位置
        body = self.tokenizer.tokenize(text)[: max_len - 2]
        tokens = ["<bos>"] + body + ["<eos>"]
        unk = self.stoi["<unk>"]
        return [self.stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        """将索引序列解码回文本，跳过 <bos>/<pad>，遇到 <eos> 停止"""
        words: List[str] = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if token in {"<bos>", "<pad>"}:
                continue
            if token == "<eos>":
                break
            words.append(token)
        return " ".join(words)
