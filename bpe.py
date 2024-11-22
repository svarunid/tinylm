import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import regex as re
from tqdm import tqdm


def _bpe_merge(word: List[bytes], pair: bytes) -> List[bytes]:
    new_word = []
    i = 0
    while i < len(word) - 1:
        if word[i] + word[i + 1] == pair:
            new_word.append(pair)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    if i < len(word):
        new_word.append(word[i])
    return new_word


def _bpe_encode(data: List[str], vocab: Dict[bytes, int], tokens: List[int]):
    words = [[bytes([b]) for b in word.encode("utf-8")] for word in data]
    for word in words:
        if len(word) == 1:
            tokens.append(vocab[word[0]])
            continue

        while True:
            min_pair, min_rank = None, None
            for i in range(len(word) - 1):
                pair = word[i] + word[i + 1]
                if pair in vocab:
                    rank = vocab[pair]
                    if min_rank is None or rank < min_rank:
                        min_pair, min_rank = pair, rank

            if min_pair is None:
                break

            word = _bpe_merge(word, min_pair)

        tokens.extend([vocab[b] for b in word])


class BytePairEncoder:
    def __init__(self, pattern: str):
        self.regex: re.Pattern = re.compile(pattern)
        self._vocab = {bytes([i]): i for i in range(2**8)}

    def encode(self, data: str, allowed_special: List[str] = []) -> List[int]:
        assert hasattr(self, "_mapping")

        tokens = []
        if hasattr(self, "special_regex"):
            start = 0
            while data:
                # Search for the next special token in data
                special_match = self.special_regex.search(data, start)
                if not special_match:
                    break

                # Extract the special token and check if it is allowed
                special_token = special_match.group()
                if special_token not in allowed_special:
                    start = special_match.start() + 1
                    continue

                # Encode the text before the special token
                text = data[: special_match.start()]
                if text:
                    _bpe_encode(re.findall(self.regex, text), self._vocab, tokens)

                # Add the special token to the tokens list
                tokens.append(self._vocab[special_token.encode("utf-8")])

                data = data[special_match.end() :]
                start = 0

        # Encode any remaining data
        if data:
            _bpe_encode(re.findall(self.regex, data), self._vocab, tokens)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        assert hasattr(self, "_mapping")

        text = b"".join(self._mapping[tok] for tok in tokens)
        return text.decode("utf-8", errors="replace")

    def train(self, vocab_size: int, data: str, special_tokens: List[str] = []):
        if vocab_size < 2**8:
            raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
        print("Training BPE. Number of character: ", len(data))

        start = time.perf_counter()
        # Split text into words based on the given pattern
        data = [[bytes([b]) for b in word.encode("utf-8")] for word in re.findall(self.regex, data)]

        pbar = tqdm(total=vocab_size - 256)
        while len(self._vocab) < vocab_size:
            # Compute the frequencies of pairs of bytes
            stats = Counter()
            for word in data:
                for pair in zip(word[:-1], word[1:]):
                    stats[pair] += 1

            if stats:
                # Find the pair with highest frequency of occurrence
                most_common_pair = max(stats, key=lambda x: stats[x])
                pair = most_common_pair[0] + most_common_pair[1]
                self._vocab[pair] = len(self._vocab)

                # Merge the most common pair and replace the occurrences
                # of the pair in the original text
                new_words = []
                for word in data:
                    new_word = _bpe_merge(word, pair)
                    new_words.append(new_word)
                data = new_words
            pbar.update(1)
        pbar.close()

        # Create a regex to match special tokens and add them to vocabulary
        if special_tokens:
            self.special_regex = re.compile(r"|".join(re.escape(tok) for tok in special_tokens))
            for tok in special_tokens:
                tok = tok.encode("utf-8")
                self._vocab[tok] = len(self._vocab)

        self._mapping = {token: pair for pair, token in self._vocab.items()}

        print(f"Total time: {(time.perf_counter() - start) / 60} min")

    def save(self, path):
        assert hasattr(self, "_mapping")

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Avoid saving _mapping to reduces file size.
        attr = vars(self)
        del attr["_mapping"]

        with file_path.open("wb+") as f:
            pickle.dump(attr, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            attr = pickle.load(f)

        obj = cls(attr["regex"])
        obj._vocab, obj._mapping = attr["_vocab"], {token: pair for pair, token in attr["_vocab"].items()}

        if "special_regex" in attr:
            obj.special_regex = attr["special_regex"]

        return obj
