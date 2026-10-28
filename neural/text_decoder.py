"""Experimental text-constrained search with an independent byte grammar.

The general LM can propose every legal token path, including partial UTF-8
and tokens crossing the committed boundary. The conditional model may guide
the finite search budget; it never determines text legality. Final ranking
uses canonical joint-text LM paths, optionally retaining legacy fusion as a
transition control. An optional backend owns LM state; text constraints and
full-vocabulary-normalized scores retain the same interface.
"""
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import heapq
import math

import torch

from byte_constraints import PinyinConstraint, ReadingIndex, TokenTrie
from decoder import Decoder
from prefix_cache import PrefixCacheLM
from token_bytes import heal_context, raw_token_bytes


@dataclass(frozen=True)
class Path:
    state: int
    ids: tuple
    raw: bytes
    q: float


class TextDecoder(Decoder):
    """Same request interface as Decoder; an opt-in research backend.

    ``guide_weight`` affects search only. ``final_mode='lm'`` uses the general
    LM plus explicit learned-reading bonuses; ``'fused'`` is a control arm.
    Partial UTF-8 frontiers have their own beam; no small-model threshold can
    rule a character out. Like any bounded beam, this is not exhaustive search.
    """

    def __init__(self, evaluator, rescorer, readings_map, beam=16,
                 per_hypothesis=24, rewind_tokens=2, guide_weight=1.0,
                 final_mode="lm", memo_limit=1024, paths_per_surface=2,
                 cache_prefix=True, cache_guide=True, cache_paths=0,
                 cache_requests=8, lm_backend=None):
        if (type(beam) is not int or beam < 1 or type(per_hypothesis) is not int
                or per_hypothesis < 1 or type(memo_limit) is not int or memo_limit < 1
                or not math.isfinite(guide_weight) or guide_weight < 0
                or final_mode not in ("lm", "fused")
                or type(paths_per_surface) is not int or paths_per_surface < 1
                or type(cache_prefix) is not bool
                or type(cache_guide) is not bool
                or type(cache_paths) is not int or cache_paths < 0
                or (cache_paths and not cache_prefix)
                or type(cache_requests) is not int or cache_requests < 0
                or type(rewind_tokens) is not int or rewind_tokens < 0):
            raise ValueError("invalid text decoder settings")
        super().__init__(evaluator, rescorer, None, beam, per_hypothesis)
        self.device = getattr(rescorer, "device", self.device)
        self.reading_index = ReadingIndex(readings_map)
        self.token_bytes = raw_token_bytes(rescorer.tokenizer)
        self.tokens = TokenTrie(self.token_bytes)
        self.rewind_tokens, self.guide_weight = rewind_tokens, guide_weight
        self.final_mode, self.memo_limit = final_mode, memo_limit
        self.paths_per_surface = paths_per_surface
        self.cache_prefix, self._prefix_lm = cache_prefix, None
        self.cache_guide, self._guide_scores = cache_guide, None
        self.cache_paths, self._branch_lm = cache_paths, None
        self.cache_requests, self._requests = cache_requests, None
        self.lm_backend = lm_backend
        self._prefix = None
        self._memo = OrderedDict()
        self.last_audit = {}
        self.last_components = []

    def _cached_model(self):
        if self._prefix_lm is None:
            self._prefix_lm = PrefixCacheLM(self.rescorer)
        return self._prefix_lm

    def lm_next(self, context, sequences):
        if self.lm_backend is not None:
            return self.lm_backend.next(context, sequences)
        if self.cache_prefix:
            if self.cache_paths:
                if self._branch_lm is None:
                    from branch_cache import BranchCacheLM
                    self._branch_lm = BranchCacheLM(self.rescorer, self._cached_model(),
                                                    max_paths=self.cache_paths)
                return self._branch_lm.next(context, sequences)
            return self._cached_model().next(context, sequences)
        return super().lm_next(context, sequences)

    def model_score(self, prompt, texts):
        if self.cache_guide:
            if self._guide_scores is None:
                from conditional_prefix import ConditionalPrefixScores
                self._guide_scores = ConditionalPrefixScores(self.evaluator)
            return self._guide_scores.score(prompt, texts)
        return super().model_score(prompt, texts)

    def lm_tails(self, context, prefixes, tails):
        if self.lm_backend is not None:
            return self.lm_backend.tails(context, prefixes, tails)
        if self.cache_prefix:
            return self._cached_model().tails(context, prefixes, tails)
        return super().lm_tails(context, prefixes, tails)

    def _next(self, prefix, sequences):
        if self._prefix != tuple(prefix):
            self._prefix, self._memo = tuple(prefix), OrderedDict()
        missing = [s for s in dict.fromkeys(sequences) if s not in self._memo]
        if missing:
            for seq, row in zip(missing, self.lm_next(list(prefix), missing).to(torch.bfloat16)):
                # A view would retain its whole batch after sibling rows are
                # evicted, defeating the LRU's memory bound during editing.
                self._memo[seq] = row.clone()
        # Keep current rows alive even if the cache budget is smaller than a
        # batch. Trimming after gathering avoids invalidating this request.
        rows = []
        for seq in sequences:
            rows.append(self._memo[seq])
            self._memo.move_to_end(seq)
        while len(self._memo) > self.memo_limit:
            self._memo.popitem(last=False)
        return torch.stack(rows).float()

    @staticmethod
    def _text(path, fixed):
        if len(path.raw) < len(fixed):
            return ""
        # The automaton guarantees valid UTF-8 prefixes. Only its unfinished
        # final character may be omitted while scoring a search frontier.
        return path.raw[len(fixed):].decode("utf-8", errors="ignore")

    def _canonical_scores(self, healed, texts, fallback):
        context = healed.original_context.decode("utf-8")
        sequences = [self.rescorer.tokenizer(context + t)["input_ids"] for t in texts]
        for text, sequence in zip(texts, sequences):
            raw = b"".join(self.token_bytes[i] or b"" for i in sequence)
            if raw != (context + text).encode("utf-8"):
                return fallback, "non_roundtripping_candidate"
        common = 0
        for values in zip(*sequences):
            if len(set(values)) != 1:
                break
            common += 1
        if not common:
            # Do not invent a BOS/newline for a no-BOS model. Search-path
            # scores still share the same exact retained context prefix.
            return fallback, "no_common_causal_prefix"
        if len(texts) == 1:
            return fallback, "single_candidate"
        scores = []
        for offset in range(0, len(sequences), 8):
            chunk = sequences[offset:offset + 8]
            scores.extend(self.lm_tails(chunk[0][:common], [()] * len(chunk),
                                        [tuple(s[common:]) for s in chunk]))
        return [score / len(text) for score, text in zip(scores, texts)], "joint_canonical"

    def decode(self, context, code, anchor=0, known=None, pins=(), learned=(),
               weight=0.0, aux=()):
        if not self.cache_requests:
            return self._decode(context, code, anchor, known, pins, learned, weight, aux)
        from request_cache import RequestCache
        if self._requests is None:
            self._requests = RequestCache(self.cache_requests)
        # Materialize optional iterables once, so the key and computation see
        # exactly the same constraints. Model weights/vocabulary are fixed for
        # the lifetime of this decoder, as for its lower-level score caches.
        pins, learned, aux = tuple(pins), tuple(learned), tuple(aux)
        key = (context, code, anchor, repr(known), repr(pins), repr(learned), weight,
               repr(aux), self.beam, self.per_hypothesis, self.rewind_tokens,
               self.guide_weight, self.final_mode, self.paths_per_surface,
               self.weight, self.memo_limit, self.cache_prefix, self.cache_guide,
               self.cache_paths)
        previous = self._requests.get(key)
        if previous is not None:
            self.last_audit, self.last_components = previous["audit"], previous["components"]
            self.last_audit.update(request_cache_hit=True, lm_rows=0, expanded_paths=0,
                                   expanded_tokens=0, frontiers=0, guide_texts=0)
            if self._guide_scores is not None:
                self.last_audit["guide_cache"] = self._guide_scores.stats.copy()
            if self._branch_lm is not None:
                self.last_audit["path_cache"] = self._branch_lm.stats.copy()
            return previous["candidates"]
        candidates = self._decode(context, code, anchor, known, pins, learned, weight, aux)
        self.last_audit["request_cache_hit"] = False
        self._requests.put(key, dict(candidates=candidates, audit=self.last_audit,
                                     components=self.last_components))
        return candidates

    @torch.inference_mode()
    def _decode(self, context, code, anchor=0, known=None, pins=(), learned=(),
                weight=0.0, aux=()):
        self.last_audit = dict(lm_rows=0, expanded_paths=0, expanded_tokens=0,
                               frontiers=0, guide_texts=0, final_mode=self.final_mode,
                               cache_prefix=self.cache_prefix, cache_guide=self.cache_guide,
                               cache_paths=self.cache_paths)
        self.last_components = []
        if not code:
            return []
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid learned-reading weight")
        try:
            healed = heal_context(self.rescorer.tokenizer, self.token_bytes, context,
                                  self.rewind_tokens)
            constraint = PinyinConstraint(self.reading_index, code, healed.fixed_prefix,
                                          pins=pins, aux=aux)
        except ValueError:
            # In an editor, incompatible normalization or stale constraints
            # retain existing dictionary results rather than changing context.
            self.last_audit["fallback"] = "incompatible_context_or_constraints"
            return []
        self.last_audit.update(actual_rewind=healed.actual_rewind,
                               fixed_prefix_bytes=len(healed.fixed_prefix),
                               constraint_states=len(constraint._edges))
        complete = len(code) // 2
        prompt = None
        if self.evaluator is not None and complete and (self.guide_weight or self.final_mode == "fused"):
            try:
                prompt, _ = self.prompts(context[-64:], code[:2*complete], anchor, known)
                if len(prompt) + constraint.syllables > self.window():
                    prompt = None
            except ValueError:
                prompt = None
        self.last_audit["conditional_active"] = prompt is not None
        conditional = {"": 0.0}

        def model_scores(paths):
            texts = [self._text(path, healed.fixed_prefix)[:complete] for path in paths]
            if prompt is None:
                return [0.0] * len(paths)
            missing = list(dict.fromkeys(t for t in texts if t not in conditional))
            for offset in range(0, len(missing), 32):
                chunk = missing[offset:offset + 32]
                conditional.update(zip(chunk, self.model_score(prompt, chunk)))
            self.last_audit["guide_texts"] += len(missing)
            return [conditional[t] for t in texts]

        def bonus(text):
            covered = set()
            for start, reading in learned:
                if type(start) is not int or start < 0:
                    raise ValueError("invalid learned-reading position")
                end = min(len(text), start + len(reading))
                if start < end and text[start:end] == reading[:end-start]:
                    covered.update(range(start, end))
            return weight * len(covered)

        initial = Path(constraint.initial_state, (), b"", 0.0)
        buckets = {constraint.progress(initial.state): [initial]}
        pending = [constraint.progress(initial.state)]
        finished = []
        while pending:
            progress = heapq.heappop(pending)
            paths = buckets.pop(progress)
            self.last_audit["frontiers"] += 1
            scores = model_scores(paths)
            ranked = sorted(zip(paths, scores), key=lambda pair: (
                self.weight * pair[0].q + self.guide_weight * pair[1]
                + bonus(self._text(pair[0], healed.fixed_prefix))), reverse=True)
            # Preserve alternative token histories inside the search, but do
            # not let many segmentations of one surface consume the entire
            # beam. At completion the budget counts distinct texts only.
            terminal = constraint.is_final(paths[0].state)
            kept, seen, surfaces = [], set(), defaultdict(int)
            for path, _ in ranked:
                if (path.ids not in seen and
                        surfaces[path.raw] < (1 if terminal else self.paths_per_surface)):
                    kept.append(path)
                    seen.add(path.ids)
                    surfaces[path.raw] += 1
                if len(kept) == self.beam:
                    break
            if terminal:
                finished = kept
                break
            sequences = [path.ids for path in kept]
            before = set(self._memo) if self._prefix == healed.prefix_ids else set()
            q = self._next(healed.prefix_ids, sequences)
            self.last_audit["lm_rows"] += sum(seq not in before for seq in sequences)
            self.last_audit["expanded_paths"] += len(kept)
            for path, row in zip(kept, q):
                allowed = self.tokens.allowed(constraint, path.state)
                if not allowed:
                    continue
                # Ignore padded model-head entries; no token representation
                # exists for them. Never normalize over only the legal IDs.
                indices = torch.tensor([token for token, _ in allowed], device=row.device)
                values = row[indices].tolist()
                groups = defaultdict(list)
                for (token, state), value in zip(allowed, values):
                    groups[constraint.progress(state)].append((value, token, state))
                for destination, choices in groups.items():
                    assert destination > progress
                    # Compare only expansions ending at the same text progress.
                    # A multi-character token never competes against a shorter
                    # fragment at this local shortlist step.
                    choices.sort(reverse=True)
                    chosen = choices[:self.per_hypothesis]
                    self.last_audit["expanded_tokens"] += len(chosen)
                    if destination not in buckets:
                        buckets[destination] = []
                        heapq.heappush(pending, destination)
                    buckets[destination].extend(Path(state, path.ids + (token,),
                        path.raw + self.token_bytes[token], path.q + value)
                        for value, token, state in chosen)

        best = {}
        for path in finished:
            assert path.raw.startswith(healed.fixed_prefix)
            text = path.raw[len(healed.fixed_prefix):].decode("utf-8")
            assert len(text) == constraint.syllables
            if text not in best or path.q > best[text].q:
                best[text] = path
        if not best:
            return []
        texts, paths = list(best), list(best.values())
        q, method = self._canonical_scores(healed, texts, [p.q / len(t) for t, p in best.items()])
        m = model_scores(paths)
        self.last_components = [dict(text=text, lm=qvalue, conditional=mvalue / len(text),
                                     fused=mvalue / len(text) + self.weight * qvalue,
                                     bonus=bonus(text) / len(text))
                                for text, qvalue, mvalue in zip(texts, q, m)]
        values = [(text, (qvalue if self.final_mode == "lm" else
                          mvalue / len(text) + self.weight * qvalue) + bonus(text) / len(text))
                  for text, qvalue, mvalue in zip(texts, q, m)]
        self.last_audit.update(final_candidates=len(values), final_scoring=method)
        if self._guide_scores is not None:
            self.last_audit["guide_cache"] = self._guide_scores.stats.copy()
        if self._branch_lm is not None:
            self.last_audit["path_cache"] = self._branch_lm.stats.copy()
        return sorted(values, key=lambda pair: pair[1], reverse=True)
