import os
from collections import Counter

from utils.constants import FILE_SEP, UNK_TOKEN, SPECIAL_TOKENS_INDEX


class IndexDictionary:
    def __init__(self, iterable=None, mode="shared", vocabulary_size=None):
        self.special_tokens = list(SPECIAL_TOKENS_INDEX.keys())
        if iterable is not None:
            # I: Not a question but a BIG NOTE. It is at this point that the integer values corresponding to the vocab
            # tokens are set.
            self.vocab_tokens, self.token_counts = self._build_vocabulary(iterable, vocabulary_size)
            self.token_index_dict = {token: index for index, token in enumerate(self.vocab_tokens)}
            self.vocabulary_size = len(self.vocab_tokens)

        self.mode = mode

    def token_to_index(self, token):
        try:
            return self.token_index_dict[token]
        except KeyError:
            return self.token_index_dict[UNK_TOKEN]

    def index_sentence(self, sentence):
        return [self.token_to_index(token) for token in sentence]

    def tokenify_indexes(self, token_indexes):
        return [self.index_to_token(token_index) for token_index in token_indexes]

    def _build_vocabulary(self, iterable, vocabulary_size):
        # Q: Why is setting of vocabulary size needed?
        counter = Counter()
        for token in iterable:
            counter[token] += 1

        if vocabulary_size is not None:
            tokens_counts = counter.most_common(vocabulary_size - len(self.special_tokens))
        else:
            tokens_counts = counter.items()

        tokens = []
        counts = []
        for token, count in tokens_counts:
            tokens.append(token)
            counts.append(count)

        vocab_tokens = self.special_tokens + tokens
        token_counts = [0] * len(self.special_tokens) + [count for token, count in counter.items()]

        return vocab_tokens, token_counts

    def save(self, data_dir):
        vocabulary_filepath = os.path.join(data_dir, f"vocabulary-{self.mode}.txt")
        with open(vocabulary_filepath, "w", encoding="utf-8") as file:
            for vocab_index, (vocab_token, count) in enumerate(
                    zip(self.vocab_tokens, self.token_counts)):
                file.write(str(vocab_index) + FILE_SEP + vocab_token + FILE_SEP + str(count) + "\n")

    @classmethod
    def load(cls, data_dir, mode="shared", vocabulary_size=None):
        vocabulary_filepath = os.path.join(data_dir, f"vocabulary-{mode}.txt")

        vocab_tokens = {}
        token_counts = []
        with open(vocabulary_filepath, encoding="utf-8") as file:
            for line in file:
                vocab_index, vocab_token, count = line.strip().split(FILE_SEP)
                vocab_tokens[int(vocab_index)] = vocab_token
                token_counts.append(int(count))

        if vocabulary_size is not None:
            vocab_tokens_copy = {}
            for k, v in vocab_token.items():
                if k > vocabulary_size:
                    break
                vocab_tokens_copy[k] = v

            vocab_tokens = vocab_tokens_copy
            token_counts = token_counts[:vocabulary_size]

        instance = cls(mode=mode)
        instance.vocab_tokens = vocab_tokens
        instance.token_counts = token_counts
        instance.token_index_dict = {token:index for index, token in vocab_tokens.items()}
        instance.vocabulary_size = len(vocab_tokens)

        return instance

    def index_to_token(self, index):
        if index >= self.vocabulary_size:
            return self.vocab_tokens[UNK_TOKEN]
        else:
            return self.vocab_tokens[index]
