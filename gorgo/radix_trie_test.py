from gorgo.radix_trie import RadixTrie


def test_insert_and_cached_prefix_length():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 4
    assert t.cached_prefix_length([1, 2, 3, 4, 5], "a") == 4
    assert t.cached_prefix_length([1, 2, 9], "a") == 2  # partial edge credit
    assert t.cached_prefix_length([1, 2, 3, 4], "b") == 0
    assert t.cached_prefix_length([], "a") == 0


def test_split_inherits_tags():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")
    # A diverging insert from "b" splits the edge; the split prefix must
    # keep "a"'s tag (every prefix of a cached sequence is cached).
    t.insert([1, 2, 7, 8], endpoint="b")
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 4
    assert t.cached_prefix_length([1, 2], "a") == 2
    assert t.cached_prefix_length([1, 2, 7, 8], "b") == 4
    assert t.cached_prefix_length([1, 2, 3], "b") == 2


def test_cached_prefix_lengths_batched_matches_single():
    t = RadixTrie()
    t.insert([1, 2, 3], endpoint="a")
    t.insert([1, 2, 3, 4, 5], endpoint="b")
    t.insert([9, 9], endpoint="c")
    seq = [1, 2, 3, 4, 5, 6]
    batched = t.cached_prefix_lengths(seq, ["a", "b", "c", "d"])
    for e in ("a", "b", "c", "d"):
        assert batched[e] == t.cached_prefix_length(seq, e)


def test_remove_endpoint():
    t = RadixTrie()
    t.insert([1, 2, 3], endpoint="a")
    t.insert([1, 2, 3], endpoint="b")
    removed = t.remove_endpoint("a")
    assert removed > 0
    assert t.cached_prefix_length([1, 2, 3], "a") == 0
    assert t.cached_prefix_length([1, 2, 3], "b") == 3  # other tags survive
    assert t.remove_endpoint("a") == 0  # idempotent
    # Sequence bookkeeping is untouched by tag removal.
    assert t.num_sequences == 2


def test_clear():
    t = RadixTrie()
    t.insert([1, 2], endpoint="a")
    t.clear()
    assert t.num_sequences == 0
    assert t.cached_prefix_length([1, 2], "a") == 0
