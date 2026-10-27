"""Small LRU for complete decoder results, with no shared mutable payloads."""
from collections import OrderedDict
from copy import deepcopy


class RequestCache:
    """Cache values under caller-supplied, immutable, hashable request keys.

    Keys use normal Python equality; this cache does not normalize inputs or
    infer which decoder settings belong in a key. Values must support deepcopy.
    Each stored value and each retrieved value owns its mutable contents, so
    callers may edit candidates, audit records, and score components safely.
    Capacity bounds entries, and zero disables storage for controlled comparisons.
    """

    def __init__(self, capacity=8):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("request cache capacity must be a nonnegative integer")
        self._capacity = capacity
        self._entries = OrderedDict()

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return len(self._entries)

    def clear(self):
        self._entries.clear()

    def get(self, key, default=None):
        """Return an independent copy on a hit, or the supplied default."""
        try:
            value = self._entries[key]
        except KeyError:
            return default
        result = deepcopy(value)
        self._entries.move_to_end(key)
        return result

    def put(self, key, value):
        """Store an independent copy, retaining the most recently used entries."""
        if self._capacity == 0:
            return
        owned = deepcopy(value)
        self._entries[key] = owned
        self._entries.move_to_end(key)
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
