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


def test_trim_endpoint_prefix_to_zero():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")
    t.insert([1, 2, 3, 4], endpoint="b")
    removed = t.trim_endpoint_prefix([1, 2, 3, 4], "a", 0)
    assert removed > 0
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 0
    assert t.cached_prefix_length([1, 2, 3, 4], "b") == 4  # other tags survive
    assert t.trim_endpoint_prefix([1, 2, 3, 4], "a", 0) == 0  # idempotent


def test_trim_endpoint_prefix_node_boundary():
    t = RadixTrie()
    t.insert([1, 2], endpoint="a")
    t.insert([1, 2, 3, 4], endpoint="a")  # nodes end at depth 2 and 4
    t.trim_endpoint_prefix([1, 2, 3, 4], "a", 2)
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 2
    assert t.cached_prefix_length([1, 2], "a") == 2  # kept: within keep_len


def test_trim_endpoint_prefix_mid_edge_is_conservative():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")  # single node, edge spans depths 1-4
    # keep_len falls mid-edge: the spanning node is untagged (no split), so
    # credit drops below keep_len rather than above it.
    t.trim_endpoint_prefix([1, 2, 3, 4], "a", 2)
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 0


def test_trim_endpoint_prefix_full_length_is_noop():
    t = RadixTrie()
    t.insert([1, 2, 3], endpoint="a")
    assert t.trim_endpoint_prefix([1, 2, 3], "a", 3) == 0
    assert t.cached_prefix_length([1, 2, 3], "a") == 3


def test_trim_endpoint_prefix_reinsert_restores_credit():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")
    t.trim_endpoint_prefix([1, 2, 3, 4], "a", 0)
    t.insert([1, 2, 3, 4], endpoint="a")  # next dispatch re-tags the path
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 4


def test_trim_endpoint_prefix_spares_branch_diverging_within_keep_len():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")
    t.insert([1, 2, 7, 8], endpoint="a")  # splits at depth 2
    t.trim_endpoint_prefix([1, 2, 3, 4], "a", 2)
    # Nodes past the divergence belong to the sibling and are never
    # visited; the shared split prefix (depth 2) is within keep_len.
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 2
    assert t.cached_prefix_length([1, 2, 7, 8], "a") == 4


def test_trim_endpoint_prefix_shared_prefix_eviction_cascades():
    t = RadixTrie()
    t.insert([1, 2, 3, 4], endpoint="a")
    t.insert([1, 2, 7, 8], endpoint="a")  # splits at depth 2
    t.trim_endpoint_prefix([1, 2, 3, 4], "a", 0)
    assert t.cached_prefix_length([1, 2, 3, 4], "a") == 0
    # Untagging the shared prefix kills the sibling's credit too. This is
    # correct: in a radix KV cache a child cannot outlive its evicted
    # parent, so an observed eviction of [1, 2] implies [1, 2, 7, 8] is
    # gone as well (the deep tag survives but is unreachable through an
    # untagged prefix).
    assert t.cached_prefix_length([1, 2, 7, 8], "a") == 0


def test_clear():
    t = RadixTrie()
    t.insert([1, 2], endpoint="a")
    t.clear()
    assert t.num_sequences == 0
    assert t.cached_prefix_length([1, 2], "a") == 0
