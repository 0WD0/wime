"""Sparse reading queries over the pinned g2pW converter, without changing it."""

from g2pw import G2PWConverter


class CorpusG2PWConverter(G2PWConverter):
    """Keep aligned results while inferring only requested character positions.

    ``__call__`` retains upstream behavior. Like the underlying converter,
    this adapter is intended for sequential use by the corpus builder.
    """

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
