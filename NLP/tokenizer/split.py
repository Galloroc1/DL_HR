from collections import defaultdict
import re
from typing import Dict, List, Tuple, Set


class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = []  # 存储合并规则

    def get_stats(self, vocab: Dict[str, int]) -> Dict[tuple, int]:
        """统计所有相邻字符对出现的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, vocab: Dict[str, int], pair: Tuple[str, str]) -> Dict[str, int]:
        """将指定的字符对在词汇表中合并"""
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}

        for word, freq in vocab.items():
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = freq

        return new_vocab

    def fit(self, corpus: List[str]) -> None:
        """训练BPE模型，考虑词频"""
        # 初始化词汇表，将每个单词拆分成字符，并统计频率
        vocab = defaultdict(int)
        for word in corpus:
            # 在每个字符之间添加空格
            spaced_word = ' '.join(list(word))
            vocab[spaced_word] += 1

        print("Initial vocabulary with frequencies:")
        for word, freq in vocab.items():
            print(f"'{word}': {freq} times")

        # 将初始字符添加到词汇表
        for word in vocab.keys():
            self.vocab.update(word.split())
        # 迭代合并最频繁的字符对
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            # 获取频率最高的字符对
            best_pair = max(pairs.items(), key=lambda x: x[1])
            # 存储合并规则为合并后的token，而不是pair
            merged_token = ''.join(best_pair[0])
            self.merge_rules.append(merged_token)
            print(f"\nMerge #{i + 1}: {best_pair[0]} (frequency: {best_pair[1]})")

            vocab = self.merge_vocab(vocab, best_pair[0])
            # 打印当前词汇表状态
            print("Current vocabulary state:")
            for word, freq in vocab.items():
                print(f"'{word}': {freq} times")
            # 将新的合并后的token添加到词汇表
            self.vocab.add(merged_token)


    def tokenize(self, text: str) -> List[str]:
        """使用训练好的BPE模型对文本进行分词"""
        # 初始化将文本拆分成字符
        tokens = ' '.join(list(text))

        # 按照学习到的合并规则顺序应用
        # 从最长的token开始尝试合并，避免子串问题
        for token in sorted(self.merge_rules, key=len, reverse=True):
            pattern = ' '.join(list(token))
            tokens = re.sub(f'(?<!\S){pattern}(?!\S)', token, tokens)

        return tokens.split()


# 示例使用
def demo_bpe():
    # 示例训练数据，包含重复单词
    corpus = [
        "low", "low", "low",  # "low" 出现3次
        "lowest",
        "newer", "newer",  # "newer" 出现2次
        "wider", "wide", "wide",  # "wide" 出现2次
        "show", "show", "show", "show"  # "show" 出现4次
    ]

    print("Training corpus:", corpus)
    print("\nWord frequencies:")
    word_freq = defaultdict(int)
    for word in corpus:
        word_freq[word] += 1
    for word, freq in word_freq.items():
        print(f"'{word}': {freq} times")

    # 初始化BPE并训练
    print("\nTraining BPE...")
    bpe = BPE(vocab_size=15)
    bpe.fit(corpus)

    # 打印最终词汇表
    print("\nFinal vocabulary:")
    print(sorted(list(bpe.vocab)))

    # 测试分词
    test_words = ["showing", "wideness", "lowering"]
    print("\nTokenization examples:")
    for word in test_words:
        tokens = bpe.tokenize(word)
        print(f"{word} -> {tokens}")


if __name__ == "__main__":
    demo_bpe()
