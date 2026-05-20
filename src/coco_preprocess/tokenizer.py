from abc import ABC, abstractmethod
import re
from typing import List


class BaseTokenizer(ABC):
    """分词器抽象基类，所有分词器需实现 tokenize 方法"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


class WordTokenizer(BaseTokenizer):
    """简单词级分词器：小写化 → 去除非字母数字字符 → 按空白切分"""

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # 将连续的非字母数字字符替换为单个空格
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = text.strip()
        if not text:
            return []
        return text.split()
