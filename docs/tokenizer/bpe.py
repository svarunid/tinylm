import regex as re
from collections import Counter
from typing import List


class SimpleBPE:
    def __init__(self, pattern: str):
        self.regex = re.compile(pattern)
        self._vocab = {bytes([i]): i for i in range(2**8)}

    def encode(self, data: str, allowed_special: List[str]):
        tokens = []

        matches = re.finditer(self.special_regex, data)
        return tokens

    def decode(self):
        pass

    def train(self, vocab_size: int, data: str, special_tokens: List[str]):
        if vocab_size < 2**8:
            raise ValueError(
                "vocab_size must be at least 256, so we can encode all bytes"
            )

        # Split text into words based on the given pattern
        words = [
            [bytes([b]) for b in word.encode("utf-8")]
            for word in re.findall(self.regex, data)
        ]

        while len(self._vocab) < vocab_size:
            stats = Counter()
            for word in words:
                for pair in zip(word[:-1], word[1:]):
                    stats[pair] += 1

            if stats:
                most_common_pair = max(stats, key=lambda x: stats[x])
                pair = most_common_pair[0] + most_common_pair[1]
                self._vocab[pair] = len(self._vocab)

                new_words = []
                for word in words:
                    new_word = []
                    i = 0
                    while i < len(word) - 1:
                        if (word[i], word[i + 1]) == most_common_pair:
                            new_word.append(pair)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    if i == len(word) - 1:
                        new_word.append(word[i])
                    new_words.append(new_word)
                words = new_words

        self.special_regex = re.compile(
            r"|".join(re.escape(tok) for tok in special_tokens)
        )
        for tok in special_tokens:
            tok = tok.encode("utf-8")
            self._vocab[tok] = len(self._vocab)

        self._mapping = {token: pair for pair, token in self._vocab.items()}
