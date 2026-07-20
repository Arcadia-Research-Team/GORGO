"""Path-compressed radix trie over uint32 token-id sequences."""

from array import array


class RadixNode:
    """One node in a path-compressed radix trie.

    ``edge`` holds the token ids on the edge leading *into* this node.
    ``count`` is the number of inserted sequences that traverse this node.
    ``terminal`` is the number of sequences that end exactly at this node.
    ``replica_endpoints`` optionally records which replica URLs (or ids) hold
    KV state for the prefix ending at this node; ``None`` means unset.
    """

    __slots__ = ("edge", "children", "count", "terminal", "replica_endpoints")

    def __init__(self, edge=None, replica_endpoints: list[str] | None = None):
        self.edge = edge if edge is not None else array("I")
        self.children: dict[int, "RadixNode"] = {}
        self.count = 0
        self.terminal = 0
        self.replica_endpoints = replica_endpoints

    def __getstate__(self):
        return {
            "edge": self.edge,
            "children": self.children,
            "count": self.count,
            "terminal": self.terminal,
            "replica_endpoints": self.replica_endpoints,
        }

    def __setstate__(self, state):
        self.edge = state["edge"]
        self.children = state["children"]
        self.count = state["count"]
        self.terminal = state["terminal"]
        self.replica_endpoints = state.get("replica_endpoints")

    def add_endpoint(self, endpoint: str) -> None:
        """Record that the prefix ending at this node is cached on ``endpoint``.

        Lazily allocates the list on first tag, and dedupes on append so a
        replica appears at most once per node no matter how many sequences
        traverse it.
        """
        if self.replica_endpoints is None:
            self.replica_endpoints = [endpoint]
        elif endpoint not in self.replica_endpoints:
            self.replica_endpoints.append(endpoint)


class RadixTrie:
    """Path-compressed radix trie over token-id sequences.

    Edges use ``array('I')`` (uint32) so each token costs 4 bytes instead of
    ~28 for a Python int object. Insertion is amortized O(len(seq)).
    """

    __slots__ = ("root", "total_tokens_inserted", "num_sequences")

    def __init__(self):
        self.root = RadixNode()
        self.total_tokens_inserted = 0
        self.num_sequences = 0

    def clear(self) -> None:
        """Drop all sequences and replica tags (empty trie, counters reset)."""
        self.root = RadixNode()
        self.total_tokens_inserted = 0
        self.num_sequences = 0

    def insert(self, seq, endpoint: str | None = None) -> None:
        """Insert ``seq`` into the trie.

        When ``endpoint`` is provided, every non-root node on the traversal
        path (newly created or existing) is tagged with it via
        ``RadixNode.add_endpoint``. The demoted child during a split keeps
        its original tags -- the inserting sequence diverges from it, so it
        does *not* acquire this endpoint.
        """
        n = len(seq)
        self.total_tokens_inserted += n
        self.num_sequences += 1
        node = self.root
        node.count += 1
        i = 0
        while True:
            if i == n:
                node.terminal += 1
                if endpoint is not None and node is not self.root:
                    node.add_endpoint(endpoint)
                return
            first = seq[i]
            child = node.children.get(first)
            if child is None:
                leaf = RadixNode(edge=array("I", seq[i:]))
                leaf.count = 1
                leaf.terminal = 1
                if endpoint is not None:
                    leaf.replica_endpoints = [endpoint]
                node.children[first] = leaf
                return
            edge = child.edge
            elen = len(edge)
            j = 1  # seq[i] == edge[0] already (first-char dispatch)
            remaining = n - i
            cap = elen if elen < remaining else remaining
            while j < cap and edge[j] == seq[i + j]:
                j += 1
            if j == elen:
                child.count += 1
                if endpoint is not None:
                    child.add_endpoint(endpoint)
                node = child
                i += j
                continue
            split = RadixNode(edge=edge[:j])
            split.count = child.count + 1
            # Inherit the demoted child's replica tags: any replica that
            # cached the longer prefix ending at ``child`` has necessarily
            # also cached the (shorter) prefix ending at ``split`` -- KV
            # caches are built token-by-token, so every prefix of a cached
            # sequence is itself cached. Without this, a split triggered by
            # a later request with a different endpoint would silently
            # "erase" the old endpoint from the split prefix.
            if child.replica_endpoints:
                split.replica_endpoints = list(child.replica_endpoints)
            child.edge = edge[j:]
            split.children[child.edge[0]] = child
            node.children[first] = split
            if endpoint is not None:
                split.add_endpoint(endpoint)
            i += j
            if i == n:
                split.terminal += 1
                return
            leaf = RadixNode(edge=array("I", seq[i:]))
            leaf.count = 1
            leaf.terminal = 1
            if endpoint is not None:
                leaf.replica_endpoints = [endpoint]
            split.children[seq[i]] = leaf
            return

    def cached_prefix_length(self, seq, endpoint: str) -> int:
        """Return the length of the longest prefix of ``seq`` that is recorded
        as cached on ``endpoint``.

        Walks the trie along ``seq`` and, at each child whose full or partial
        edge matches, consults ``replica_endpoints``: if the child is tagged
        with ``endpoint`` we extend the cached count by however many tokens
        of its edge we matched; otherwise we stop. Partial-edge matches are
        still credited because KV caches are incremental, so any prefix of a
        cached sequence is also cached.
        """
        n = len(seq)
        if n == 0:
            return 0
        node = self.root
        i = 0
        cached = 0
        while i < n:
            child = node.children.get(seq[i])
            if child is None:
                break
            edge = child.edge
            elen = len(edge)
            remaining = n - i
            cap = elen if elen < remaining else remaining
            j = 1  # seq[i] == edge[0] by dispatch
            while j < cap and edge[j] == seq[i + j]:
                j += 1
            eps = child.replica_endpoints
            if eps is None or endpoint not in eps:
                break
            cached = i + j
            if j < elen:
                # Partial edge match -- no further descent possible.
                break
            node = child
            i += j
        return cached

    def cached_prefix_lengths(self, seq, endpoints) -> dict[str, int]:
        """Batched version of :meth:`cached_prefix_length` over many endpoints.

        Walks the trie along ``seq`` exactly once and maintains a live set of
        "still-qualifying" endpoints: at each node, endpoints not tagged on
        that node are frozen at their current cached count and dropped from
        further consideration. This is O(len(seq) + sum of replica_endpoints
        list sizes along the walk) instead of O(E * len(seq)).
        """
        remaining_endpoints = {e for e in endpoints}
        result: dict[str, int] = {e: 0 for e in remaining_endpoints}
        n = len(seq)
        if n == 0 or not remaining_endpoints:
            return result
        node = self.root
        i = 0
        while i < n and remaining_endpoints:
            child = node.children.get(seq[i])
            if child is None:
                break
            edge = child.edge
            elen = len(edge)
            remaining_tokens = n - i
            cap = elen if elen < remaining_tokens else remaining_tokens
            j = 1
            while j < cap and edge[j] == seq[i + j]:
                j += 1
            eps = child.replica_endpoints
            # Endpoints tagged on this child stay alive and have their
            # cached count advanced. Others freeze at their previous value.
            if eps:
                tagged = remaining_endpoints & set(eps)
            else:
                tagged = set()
            for e in tagged:
                result[e] = i + j
            remaining_endpoints = tagged
            if j < elen:
                break
            node = child
            i += j
        return result

    def remove_endpoint(self, endpoint: str) -> int:
        """Drop every ``endpoint`` tag in the trie; returns tags removed.

        Used when a replica leaves the fleet: its KV cache is gone, so
        prefixes tagged with it must stop counting as cached there. The
        nodes themselves are kept -- other replicas' tags and the
        sequence counts stay valid. O(nodes); replica churn is rare
        relative to inserts, so a full sweep is acceptable.
        """
        removed = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            eps = node.replica_endpoints
            if eps and endpoint in eps:
                eps.remove(endpoint)
                removed += 1
                if not eps:
                    node.replica_endpoints = None
            stack.extend(node.children.values())
        return removed

    def unique_token_count(self) -> int:
        """Total length of all compressed edges = KV-cache footprint after
        perfect prefix sharing."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            total += len(node.edge)
            stack.extend(node.children.values())
        return total

    def node_count(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count
