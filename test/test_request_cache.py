"""CPU-only tests for exact request reuse and isolation of decoder results."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from request_cache import RequestCache  # noqa: E402


class RequestCaching(unittest.TestCase):
    def test_complete_request_keys_match_without_prefix_or_constraint_reuse(self):
        cache = RequestCache()
        key = ("新杉田车", "vj", 2, None, ((0, "站"),), (), 0.0,
               ((0, "nz"),), (16, 24, 2, 1.0, "lm"))
        cache.put(key, ["站"])
        self.assertEqual(cache.get(tuple(list(key))), ["站"])
        changed_fields = ("新杉田", "v", 1, ("zhan",), (), ("战",), 0.5,
                          (), (16, 24, 2, 1.0, "fused"))
        missing = object()
        for index, changed in enumerate(changed_fields):
            with self.subTest(field=index):
                other = list(key)
                other[index] = changed
                self.assertIs(cache.get(tuple(other), missing), missing)

    def test_hash_collisions_do_not_reuse_an_unequal_key(self):
        class CollidingKey(str):
            def __hash__(self):
                return 42

        cache = RequestCache()
        cache.put(CollidingKey("first"), ["first"])
        cache.put(CollidingKey("second"), ["second"])
        self.assertEqual(cache.get(CollidingKey("first")), ["first"])
        self.assertEqual(cache.get(CollidingKey("second")), ["second"])
        self.assertIsNone(cache.get(CollidingKey("third")))

    def test_default_capacity_is_eight(self):
        cache = RequestCache()
        self.assertEqual(cache.capacity, 8)
        for index in range(9):
            cache.put(index, index)
        self.assertEqual(len(cache), 8)
        self.assertIsNone(cache.get(0))
        self.assertEqual([cache.get(index) for index in range(1, 9)], list(range(1, 9)))

    def test_hit_refreshes_lru_but_miss_does_not(self):
        cache = RequestCache(capacity=2)
        cache.put("first", 1)
        cache.put("second", 2)
        self.assertEqual(cache.get("first"), 1)
        self.assertIsNone(cache.get("missing"))
        cache.put("third", 3)
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), 1)
        self.assertEqual(cache.get("third"), 3)

    def test_replacement_refreshes_lru_without_consuming_an_extra_entry(self):
        cache = RequestCache(capacity=2)
        cache.put("first", 1)
        cache.put("second", 2)
        cache.put("first", 10)
        self.assertEqual(len(cache), 2)
        cache.put("third", 3)
        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), 10)

    def test_store_and_retrieval_isolate_nested_candidates_audit_and_components(self):
        cache = RequestCache()
        source = {
            "candidates": [["你好", -0.5]],
            "audit": {"constraints": {"pins": [[0, "你"]]}, "paths": [1, 2]},
            "components": [{"text": "你好", "scores": {"lm": -1.0}, "tokens": [3, 4]}],
        }
        expected = deepcopy(source)
        cache.put("request", source)
        cache.put("another-request", source)
        source["candidates"][0][0] = "拟好"
        source["audit"]["constraints"]["pins"][0][1] = "拟"
        source["components"][0]["scores"]["lm"] = 0.0
        first = cache.get("request")
        self.assertEqual(first, expected)
        first["candidates"][0][1] = 100.0
        first["audit"]["paths"].clear()
        first["audit"]["constraints"]["pins"].append([1, "好"])
        first["components"][0]["scores"]["lm"] = 100.0
        first["components"][0]["tokens"].append(5)
        self.assertEqual(cache.get("request"), expected)
        self.assertEqual(cache.get("another-request"), expected)

    def test_none_and_empty_values_are_hits(self):
        cache = RequestCache()
        missing = object()
        cache.put("none", None)
        cache.put("empty", [])
        self.assertIsNone(cache.get("none", missing))
        self.assertEqual(cache.get("empty", missing), [])
        self.assertIs(cache.get("missing", missing), missing)

    def test_zero_capacity_and_clear_leave_no_cached_results(self):
        disabled = RequestCache(capacity=0)
        disabled.put("request", ["你好"])
        self.assertEqual(len(disabled), 0)
        self.assertIsNone(disabled.get("request"))
        cache = RequestCache(capacity=1)
        cache.put("request", ["你好"])
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get("request"))
        cache.put("next", ["再见"])
        self.assertEqual(cache.get("next"), ["再见"])

    def test_invalid_capacity_is_rejected(self):
        for capacity in (-1, True, False, 1.5, "8", None):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                RequestCache(capacity=capacity)

    def test_failed_copy_does_not_replace_or_evict_an_existing_result(self):
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise RuntimeError("cannot copy")

        cache = RequestCache(capacity=1)
        cache.put("request", ["你好"])
        for key in ("request", "other"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "cannot copy"):
                cache.put(key, Uncopyable())
            self.assertEqual(len(cache), 1)
            self.assertEqual(cache.get("request"), ["你好"])


if __name__ == "__main__":
    unittest.main()
