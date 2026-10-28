"""Sparse reading queries over the pinned g2pW converter, without changing it."""

from functools import lru_cache

from g2pw import G2PWConverter


class CachedTokenizer:
    """Cache default tokenization for a converter's fixed tokenizer vocabulary.

    g2pW tokenizes each word again for every queried character's context. Keep
    bounded immutable results per converter and give callers fresh lists.
    Calls that pass tokenizer options retain the tokenizer's original behavior.
    """

    def __init__(self, tokenizer, maxsize=32768):
        self.tokenizer = tokenizer
        self._tokens = lru_cache(maxsize=maxsize)(lambda text: tuple(tokenizer.tokenize(text)))

    def tokenize(self, text, *args, **kwargs):
        if args or kwargs or not isinstance(text, str):
            return self.tokenizer.tokenize(text, *args, **kwargs)
        return list(self._tokens(text))

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class CorpusG2PWConverter(G2PWConverter):
    """Keep aligned results while inferring only requested character positions.

    ``__call__`` retains upstream behavior. Like the underlying converter,
    this adapter is intended for sequential use by the corpus builder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = CachedTokenizer(self.tokenizer)

    def read_positions(self, sentences, positions):
        sentences, positions = list(sentences), list(positions)
        if len(sentences) != len(positions):
            raise ValueError("one position list is required per sentence")
        requested = []
        for sentence, indices in zip(sentences, positions):
            indices = frozenset(indices)
            if any(type(i) is not int or not 0 <= i < len(sentence) for i in indices):
                raise ValueError("reading positions must be character indices within their sentence")
            requested.append(indices)
        # Upstream TextDataset lowercases text without remapping query_id.
        # Non-Han characters such as İ expand during lowercasing and would
        # shift every later Han query. Replace them with one space only in
        # the model input; stored corpus text and all character indices stay
        # intact. This deliberately loses that non-Han context character.
        sentences = ["".join(char if len(char.lower()) == 1 else " " for char in sentence)
                     for sentence in sentences]
        if getattr(self, "_requested_positions", None) is not None:
            raise RuntimeError("g2pW positional reads cannot be nested")
        self._requested_positions = requested
        try:
            return super().__call__(sentences)
        finally:
            self._requested_positions = None

    def _prepare_data(self, sentences):
        texts, query_ids, sent_ids, partial_results = super()._prepare_data(sentences)
        requested = getattr(self, "_requested_positions", None)
        if requested is None:
            return texts, query_ids, sent_ids, partial_results
        if len(sentences) != len(requested):
            raise ValueError("g2pW changed the positional query batch size")
        selected = [(text, query, sent) for text, query, sent in zip(texts, query_ids, sent_ids)
                    if query in requested[sent]]
        if selected:
            texts, query_ids, sent_ids = map(list, zip(*selected))
        else:
            texts, query_ids, sent_ids = [], [], []
        return texts, query_ids, sent_ids, partial_results
