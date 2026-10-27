"""Text legality for double pinyin, independent of model tokens and scores.

Each output position is a set of dictionary characters. A deterministic byte
graph concatenates their UTF-8 tries, optionally preceded by exact committed
bytes that a decoder must regenerate while healing its token boundary. No
cartesian product of complete strings is materialized. TokenTrie intersects
this graph with raw token bytes, including tokens ending inside a character.
"""
from collections import defaultdict
from weakref import WeakKeyDictionary


KEYS = frozenset("abcdefghijklmnopqrstuvwxyz;")
AUX_KEYS = frozenset("abcdefghijklmnopqrstuvwxyz")


class ReadingIndex:
    """Reusable immutable reading lookup; build once for a loaded dictionary."""

    def __init__(self, readings_map):
        by_code = defaultdict(list)
        for char, readings in readings_map.items():
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("dictionary entries must be individual characters")
            try:
                encoded = char.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("dictionary character is not valid Unicode") from error
            for code, groups in readings.items():
                if (not isinstance(code, str) or len(code) != 2
                        or any(key not in KEYS for key in code)):
                    raise ValueError("dictionary reading must be a double-pinyin syllable")
                by_code[code].append((char, encoded, tuple(groups)))
        self.by_code = {code: tuple(items) for code, items in by_code.items()}
        partial = defaultdict(list)
        for code, items in self.by_code.items():
            partial[code[0]].extend(items)
        self.partial = {code: tuple(items) for code, items in partial.items()}

    def get(self, keys):
        return (self.partial if len(keys) == 1 else self.by_code).get(keys, ())


class PinyinConstraint:
    """A finite byte language with integer states and no probability gates.

    The language is ``fixed_prefix`` followed by exactly one character per
    complete syllable, plus one provisional character for a trailing half
    syllable. Pins and auxiliary codes intersect these character sets. Empty
    intersections make the whole language empty, including all token prefixes.
    ``fixed_prefix`` is raw bytes: it may itself begin inside a UTF-8 character
    when the retained model prefix ended inside that same character.
    """

    def __init__(self, readings_map, code, fixed_prefix=b"", pins=(), aux=()):
        if not isinstance(code, str) or any(key not in KEYS for key in code):
            raise ValueError("expected lower-case double-pinyin keys")
        if not isinstance(fixed_prefix, bytes):
            raise ValueError("fixed_prefix must contain raw bytes")
        self.code, self.fixed_prefix = code, fixed_prefix
        self.syllables = (len(code) + 1) // 2
        index = readings_map if isinstance(readings_map, ReadingIndex) else ReadingIndex(readings_map)
        pinned, auxiliary, conflict = {}, {}, False
        for position, char in pins:
            self._validate_position(position)
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("each pin must be one character")
            try:
                char.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("pin is not valid Unicode") from error
            if position in pinned and pinned[position] != char:
                conflict = True
            pinned[position] = char
        for position, keys in aux:
            self._validate_position(position)
            if (not isinstance(keys, str) or not keys
                    or any(key not in AUX_KEYS for key in keys)):
                raise ValueError("auxiliary keys must be nonempty lower-case letters")
            previous = auxiliary.get(position, "")
            if not (keys.startswith(previous) or previous.startswith(keys)):
                conflict = True
            auxiliary[position] = max((previous, keys), key=len)

        alphabets = []
        for position in range(self.syllables):
            keys = code[2 * position:2 * position + 2]
            typed = auxiliary.get(position)
            # A polyphonic character can occur under several matching codes
            # for a half syllable. Aux must match one of those same readings.
            allowed = {raw for char, raw, groups in index.get(keys)
                       if (position not in pinned or char == pinned[position])
                       and (typed is None or any(group.startswith(typed) for group in groups))}
            alphabets.append(allowed)

        self._edges, self._completed, self._progress = [], [], []
        self.max_bytes = len(fixed_prefix) + sum(max(map(len, chars), default=0) for chars in alphabets)
        if conflict or any(not chars for chars in alphabets):
            self.initial_state = self._node(0, literal_offset=0 if fixed_prefix else None)
            self._final_state = None
            return

        self._final_state = self._node(self.syllables)
        following = self._final_state
        for position in range(self.syllables - 1, -1, -1):
            root = self._node(position)
            for raw in sorted(alphabets[position]):
                state = root
                for depth, byte in enumerate(raw[:-1], 1):
                    next_state = self._edges[state].get(byte)
                    if next_state is None:
                        next_state = self._node(position, within_character=depth)
                        self._edges[state][byte] = next_state
                    state = next_state
                # UTF-8 encodings of individual Unicode scalars are prefix
                # free, so all leaf edges can join the next position's root.
                self._edges[state][raw[-1]] = following
            following = root
        for offset in range(len(fixed_prefix) - 1, -1, -1):
            byte = fixed_prefix[offset]
            state = self._node(0, literal_offset=offset)
            self._edges[state][byte] = following
            following = state
        self.initial_state = following

    def _validate_position(self, position):
        if type(position) is not int or not 0 <= position < self.syllables:
            raise ValueError("constraint position is outside the composition")

    def _node(self, completed, within_character=0, literal_offset=None):
        state = len(self._edges)
        self._edges.append({})
        self._completed.append(completed)
        self._progress.append((completed + 1, within_character) if literal_offset is None
                              else (0, literal_offset))
        return state

    def step(self, state, raw):
        """Consume raw bytes, or return None when they leave the language."""
        if not isinstance(raw, bytes):
            raise ValueError("step expects raw bytes")
        if type(state) is not int or not 0 <= state < len(self._edges):
            raise ValueError("invalid constraint state")
        for byte in raw:
            state = self._edges[state].get(byte)
            if state is None:
                return None
        return state

    def is_final(self, state):
        return type(state) is int and self._final_state is not None and state == self._final_state

    def completed_characters(self, state):
        """Count newly composed characters, excluding the committed prefix."""
        if type(state) is not int or not 0 <= state < len(self._completed):
            raise ValueError("invalid constraint state")
        return self._completed[state]

    def progress(self, state):
        """A strictly increasing key for synchronizing byte-search frontiers.

        Literal states are ``(0, literal_bytes_consumed)``. Composing states
        are ``(completed_characters + 1, bytes_inside_current_character)``;
        character boundaries have a zero second component. A four-byte
        character can therefore meet a three-byte one at the same subsequent
        boundary without giving that shared node inconsistent progress.
        """
        if type(state) is not int or not 0 <= state < len(self._progress):
            raise ValueError("invalid constraint state")
        return self._progress[state]


class TokenTrie:
    """Raw token bytes, intersected lazily with a PinyinConstraint.

    None marks special/unavailable tokens. Empty tokens are excluded, so every
    accepted token consumes at least one byte and cannot create a loop. Cache
    entries are owned by weak references to each request's constraint graph.
    Duplicate byte strings retain every distinct token ID.
    """

    def __init__(self, token_bytes):
        self.token_bytes = tuple(token_bytes)
        self._edges, self._terminals = [{}], [[]]
        self.max_token_bytes = 0
        for token, raw in enumerate(self.token_bytes):
            if raw is None:
                continue
            if not isinstance(raw, bytes):
                raise ValueError("token representations must be bytes or None")
            if not raw:
                continue
            self.max_token_bytes = max(self.max_token_bytes, len(raw))
            state = 0
            for byte in raw:
                following = self._edges[state].get(byte)
                if following is None:
                    following = len(self._edges)
                    self._edges[state][byte] = following
                    self._edges.append({})
                    self._terminals.append([])
                state = following
            self._terminals[state].append(token)
        self._cache = WeakKeyDictionary()

    def allowed(self, constraint, state):
        """Return [(token_id, next_state)] without scanning the whole vocab."""
        # Validate before looking up the cache; bool must not alias state 0/1.
        constraint.step(state, b"")
        cached = self._cache.setdefault(constraint, {})
        if state not in cached:
            result, pending = [], [(0, state)]
            while pending:
                token_state, text_state = pending.pop()
                result.extend((token, text_state) for token in self._terminals[token_state])
                token_edges = self._edges[token_state]
                text_edges = constraint._edges[text_state]
                # Usually the text automaton has only one or a few branches
                # while the root of the vocabulary trie has many.
                if len(text_edges) <= len(token_edges):
                    pending.extend((token_edges[byte], following)
                                   for byte, following in text_edges.items() if byte in token_edges)
                else:
                    pending.extend((following, text_edges[byte])
                                   for byte, following in token_edges.items() if byte in text_edges)
            cached[state] = tuple(sorted(result))
        return list(cached[state])
