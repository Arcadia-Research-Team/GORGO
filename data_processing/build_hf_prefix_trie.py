"""Build radix tries over Hugging Face ``save_to_disk`` datasets (LMSYS-Chat-1M,
WildChat-4.8M, etc.) and report KV-style prefix overlap (intra-user vs global),
same metrics as ``build_prefix_trie.py``.

Expects::

    <root>/dataset_dict.json
    <root>/train/data-*.arrow

Tokenization matches ``build_eval_dataset`` (tiktoken ``gpt-4o``; message ``content``
concatenated in order). LMSYS / WildChat expose a ``conversation`` column (role/content
list): trie ingest and per-key dedup use **only** that column when present. The winning
row per ``conversation_id`` / ``conversation_hash`` is the one with the **longest**
tiktoken sequence (tie: earliest row). Splits without ``conversation`` fall back to the
first non-empty among ``messages`` / ``conversations``, with winner = max **message
count**.

**Content-hash prefix dedup** (``--dedup-content-prefix-sha256``): build a trie on
**SHA256(UTF-8 content)** per message (order preserved; role ignored). Rows are comparable
as prefixes iff digests match element-wise up to ``min(len)``; only **maximal** rows
(longest digest list in each prefix chain) are inserted into the radix trie. With this
mode, **tokens-per-conversation** stats use only those same ingested rows (no extra pass
or tiktoken for conversation-key winners); row-count / duplicate metrics still use all
rows with a key.
**WildChat default:** this mode is **on by default** when the resolved disk path looks
like WildChat or you pass ``--preset wildchat`` (including custom ``--dataset-disk-path``).
Use ``--no-dedup-content-prefix-sha256`` for the legacy longest-row-per-conversation-key
behavior. LMSYS and other roots default to **off** unless you pass the dedup flag.
When the split has ``hashed_ip`` (e.g. WildChat), prefix chains are resolved **per IP
bucket** so only same-user rows compete—smaller tries and the same assumption as
cross-row shared prefixes coming from one user.

**No dedup** (``--ingest-all-rows``): bypass content-hash prefix dedup **and** the
per-conversation winner pass; every row is tokenized and inserted. Use this to get
A/B/C numbers directly comparable to ``build_prefix_trie.py`` (same "one sequence per
row, no dedup" semantics). Still differs from that script in tokenizer (tiktoken
``gpt-4o``) and in user/group key selection.

**User key:** tries ``user_id``, ``hashed_user_id``, ``hashed_ip`` (WildChat),
``user_hash``, ``ip_hash``, ``conversation_id``, ``conversation_hash``. Override with
``--user-key-column``. If nothing matches, falls back to ``row:<index>`` (intra-user
overlap ~0).

**Conversation stats:** if the split has ``conversation_id`` or ``conversation_hash``,
also reports duplicate conversation keys (rows with count > 1), rows-per-conversation
distribution, and tokens-per-conversation for the winning row per key (not summed
across duplicate rows). Aggregates are included in checkpoints; old checkpoints without
them skip this block until you delete ``prefix_trie_checkpoints/`` and rerun from row 0.

After a full pass, per-conversation aggregates are written to
``prefix_trie_conversation_aggregates.pkl`` and
``prefix_trie_checkpoints/conversation_aggregates.pkl`` (small files, no tries).
``--conversation-stats-only`` loads small aggregate pickles only (does **not** unpickle
``rows_*.pkl``). If those are missing, scans the split with tiktoken and recomputes
conversation aggregates (slower, works with old trie checkpoints that lack
``conversation_*`` in the pickle). Written caches speed up the next stats-only run.

Volumes: ``GORGO-lmsys-chat-1m`` at ``/lmsys``, ``GORGO-hf-datasets`` at ``/datasets``.

Examples::

    modal run --env=GORGO data_processing/build_hf_prefix_trie.py::prefix_trie
    modal run --detach --env=GORGO data_processing/build_hf_prefix_trie.py::prefix_trie --preset wildchat
    modal run data_processing/build_hf_prefix_trie.py::prefix_trie --preset lmsys \\
        --dataset-disk-path /lmsys/lmsys-chat-1m
    # ``--preset auto`` tries LMSYS paths first, then WildChat. Override search order with
    # ``--extra-candidates /path/one,/path/two`` (tried before preset list).
"""

from __future__ import annotations

import os
import re
import sys
from array import array

import modal

from app import app, hf_datasets_volume, lmsys_chat_1m_volume
from build_eval_dataset import _content_to_str
from gorgo.radix_trie import RadixTrie

image = (
    modal.Image.debian_slim()
    .pip_install("datasets>=3.0", "pyarrow", "tiktoken")
    .add_local_python_source("app", "build_eval_dataset", "gorgo")
)

USER_KEY_CANDIDATES = (
    "user_id",
    "hashed_user_id",
    "hashed_ip",
    "user_hash",
    "ip_hash",
    "conversation_id",
    "conversation_hash",
)

# For duplicate / per-conversation token stats (not the same ordering as USER_KEY_CANDIDATES).
CONVERSATION_KEY_CANDIDATES = ("conversation_id", "conversation_hash")

MESSAGE_COLUMNS = ("conversation", "messages", "conversations")

# Tried in order until ``dataset_dict.json`` is found (first match wins).
LMSYS_DISK_CANDIDATES: list[tuple[str, str]] = [
    ("lmsys", "/lmsys/lmsys-chat-1m"),
    ("lmsys", "/lmsys/datasets/lmsys__lmsys-chat-1m"),
    # ``download_hf_dataset`` writes ``datasets/<hub_id>`` on the volume → ``/datasets/datasets/...``.
    ("hf", "/datasets/datasets/lmsys__lmsys-chat-1m"),
    ("hf", "/datasets/lmsys-chat-1m"),
]

# Hub download layout on ``GORGO-hf-datasets``: ``datasets/allenai__WildChat-4.8M/`` (checkpoints
# under ``.../prefix_trie_checkpoints/``). Mount volume at ``/datasets`` → use ``/datasets/datasets/...``.
WILDCHAT_DISK_CANDIDATES: list[tuple[str, str]] = [
    ("hf", "/datasets/datasets/allenai__WildChat-4.8M"),
    ("hf", "/datasets/allenai__WildChat-4.8M"),
]

CONVERSATION_AGGREGATES_FILENAME = "prefix_trie_conversation_aggregates.pkl"
# Small pickle next to ``rows_*.pkl`` (no tries); fast path for ``--conversation-stats-only``.
CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME = "conversation_aggregates.pkl"
CONVERSATION_AGGREGATES_VERSION = 6
# Bumped when trie ingest or per-conversation token semantics change (v2: one row per
# conversation key, the longest by message count; v3: ingest accounting in aggregate
# pickles; trie checkpoint schema 3 adds persisted ingest counters on rows_*.pkl;
# v4: when ``conversation`` column exists, messages and winner selection use that
# column only; winner is longest by tiktoken length, not message count;
# v5: with ``--dedup-content-prefix-sha256``, conversation token totals come only from
# prefix-ingested rows (no separate conversation-key winner pass);
# v6: content-hash dedup may partition by ``hashed_ip`` when present (checkpoint field
# ``dhfpt_content_hash_partition``).
TRIE_CHECKPOINT_SCHEMA_VERSION = 4
# Progress logs while computing overlap stats over per-user tries (can be millions of keys).
OVERLAP_STATS_LOG_EVERY_USERS = 50_000
# Log sort progress inside ``_summarize_int_distribution`` when this many values or more.
DISTRIBUTION_SORT_LOG_MIN = 300_000
# How many ``rows_*.pkl`` files to try (after exact name; rest by largest file size first).
ROWS_CHECKPOINT_MAX_LOAD_ATTEMPTS = 15
# If this basename exists under ``prefix_trie_checkpoints/``, use **only** that file for
# conversation aggregates from rows checkpoints (WildChat train full run). Remove when
# generic resolution is enough.
HARDCODED_ROWS_CHECKPOINT_BASENAME = "rows_03199860.pkl"


def _fmt_pct(num: float, den: float) -> str:
    if den <= 0:
        return "  n/a"
    return f"{100.0 * num / den:6.2f}%"


def _is_hf_save_dir(path: str) -> bool:
    """True if ``path`` looks like ``datasets`` ``save_to_disk`` output."""
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "dataset_dict.json")):
        return True
    # Single-split ``Dataset.save_to_disk`` (no dataset_dict.json).
    if os.path.isfile(os.path.join(path, "dataset_info.json")) and os.path.isfile(
        os.path.join(path, "state.json")
    ):
        return True
    return False


def _normalize_hf_disk_root(path: str) -> str:
    """Resolve path to a loadable HF on-disk dataset; fix ``.../train`` → parent when needed."""
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        raise RuntimeError(
            f"Dataset path is not a directory: {path!r}. "
            "Check Modal volume mounts: LMSYS is often /lmsys/lmsys-chat-1m; "
            "HF hub downloads on this volume use /datasets/datasets/<org>__<name> "
            "(volume-relative datasets/<org>__<name>)."
        )
    if _is_hf_save_dir(path):
        return path
    parent = os.path.dirname(path)
    if parent and _is_hf_save_dir(parent):
        print(f"[prefix_trie] Using parent with dataset metadata: {parent}")
        return parent
    hint = ""
    if path.startswith("/datasets/") and "lmsys" in path.lower():
        hint = (
            " For LMSYS-Chat-1M, data is usually on volume GORGO-lmsys-chat-1m "
            "at /lmsys/lmsys-chat-1m (not on GORGO-hf-datasets unless you ran "
            "download_hf_cli for that dataset)."
        )
    try:
        listing = sorted(os.listdir(path))[:40]
    except OSError:
        listing = []
    raise RuntimeError(
        f"Not a Hugging Face datasets save_to_disk directory: {path!r}.{hint} "
        "Expected dataset_dict.json (DatasetDict) or dataset_info.json + state.json (Dataset). "
        f"Listing: {listing}"
    )


def _find_disk_root(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for tag, path in candidates:
        if os.path.isdir(path) and _is_hf_save_dir(path):
            return tag, path
    paths = [p for _, p in candidates]
    raise RuntimeError("Could not find HF save_to_disk root. Tried: " + ", ".join(paths))


def _candidates_for_preset(preset: str) -> list[tuple[str, str]]:
    p = preset.lower().strip().replace("-", "_")
    if p in ("auto", "all", "combined"):
        return list(LMSYS_DISK_CANDIDATES) + list(WILDCHAT_DISK_CANDIDATES)
    if p in ("lmsys", "lmsys_chat", "lmsys_chat_1m"):
        return list(LMSYS_DISK_CANDIDATES)
    if p in ("wildchat", "wild_chat", "allenai_wildchat"):
        return list(WILDCHAT_DISK_CANDIDATES)
    raise ValueError(f"unknown --preset {preset!r}; use auto, lmsys, or wildchat")


def _is_wildchat_disk_root(root: str) -> bool:
    return "wildchat" in os.path.normpath(root).lower()


def _preset_is_wildchat(preset: str | None) -> bool:
    if not preset:
        return False
    p = preset.lower().strip().replace("-", "_")
    return p in ("wildchat", "wild_chat", "allenai_wildchat")


def _default_dedup_content_prefix_sha256(root: str, run_preset: str | None) -> bool:
    return _is_wildchat_disk_root(root) or _preset_is_wildchat(run_preset)


def _merge_disk_candidates(
    preset: str,
    extra_candidates: str | None,
) -> list[tuple[str, str]] | None:
    """Paths from ``extra_candidates`` (comma-separated) are tried first, then ``preset`` list."""
    merged: list[tuple[str, str]] = []
    if extra_candidates:
        for i, raw in enumerate(extra_candidates.split(",")):
            path = raw.strip()
            if path:
                merged.append((f"extra{i}", path))
    merged.extend(_candidates_for_preset(preset))
    return merged


def _stats_basename(root: str) -> str:
    base = os.path.basename(os.path.normpath(root))
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base)
    return safe or "dataset"


def _messages_from_row(
    row: dict,
    *,
    message_source: str | None = None,
) -> list:
    """Message list for tokenization / dedup.

    If ``message_source`` is set (e.g. ``conversation``), only that column is
    used. Otherwise the first non-empty among ``MESSAGE_COLUMNS`` is used.
    """
    import json

    keys = (message_source,) if message_source else MESSAGE_COLUMNS
    for key in keys:
        if key not in row:
            continue
        conv = row[key]
        if conv is None:
            continue
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except json.JSONDecodeError:
                continue
        if isinstance(conv, list):
            return conv
    return []


def _dedup_message_source(columns: set[str]) -> str | None:
    """Prefer canonical ``conversation`` list (role/content) when the column exists."""
    if "conversation" in columns:
        return "conversation"
    return None


def _dedup_stats_labels(
    message_source: str | None,
) -> tuple[str, str]:
    """JSON semantic key and console caption for per-conversation token stats."""
    if message_source == "conversation":
        return (
            "longest_row_by_tiktoken_gpt4o_on_conversation_column",
            (
                "Tokens / conversation (tiktoken gpt-4o, longest row per key by length "
                "on column 'conversation'):  "
            ),
        )
    return (
        "longest_row_by_message_count_first_matching_message_column",
        ("Tokens / conversation (tiktoken gpt-4o, longest row per key by message count):  "),
    )


def _conversation_key_column(columns: set[str]) -> str | None:
    for k in CONVERSATION_KEY_CANDIDATES:
        if k in columns:
            return k
    return None


def _conversation_key_value(row: dict, conv_col: str | None) -> str | None:
    if not conv_col:
        return None
    v = row.get(conv_col)
    if v is None or str(v) == "":
        return None
    return str(v)


def _conversation_aggregates_path(root: str) -> str:
    return os.path.join(root, CONVERSATION_AGGREGATES_FILENAME)


def _conversation_aggregates_checkpoint_path(root: str) -> str:
    return os.path.join(
        root, "prefix_trie_checkpoints", CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME
    )


def _try_load_conversation_aggregates(
    path: str,
    *,
    limit: int,
    n_total: int,
    min_sequence_len: int,
    conv_col: str,
) -> tuple[dict[str, int], dict[str, int], int, int] | None:
    import pickle

    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError):
        return None
    if payload.get("version") != CONVERSATION_AGGREGATES_VERSION:
        return None
    if payload.get("limit") != limit:
        return None
    if payload.get("n_total") != n_total:
        return None
    if payload.get("min_sequence_len") != min_sequence_len:
        return None
    if payload.get("conversation_key_column") != conv_col:
        return None
    done_row = payload.get("completed_next_row", limit)
    if done_row != limit:
        return None
    crc = payload.get("conversation_row_counts")
    ctt = payload.get("conversation_token_totals")
    if not isinstance(crc, dict) or not isinstance(ctt, dict):
        return None
    rm = payload.get("rows_missing_conversation_key")
    rs = payload.get("rows_skipped_duplicate_conversation")
    if not isinstance(rm, int) or not isinstance(rs, int):
        return None
    return crc, ctt, rm, rs


def _save_conversation_aggregates(
    path: str,
    *,
    limit: int,
    n_total: int,
    min_sequence_len: int,
    conv_col: str,
    conversation_row_counts: dict[str, int],
    conversation_token_totals: dict[str, int],
    rows_missing_conversation_key: int,
    rows_skipped_duplicate_conversation: int,
    completed_next_row: int | None = None,
) -> None:
    import pickle

    done = limit if completed_next_row is None else completed_next_row
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "version": CONVERSATION_AGGREGATES_VERSION,
                "limit": limit,
                "n_total": n_total,
                "min_sequence_len": min_sequence_len,
                "conversation_key_column": conv_col,
                "completed_next_row": done,
                "conversation_row_counts": conversation_row_counts,
                "conversation_token_totals": conversation_token_totals,
                "rows_missing_conversation_key": rows_missing_conversation_key,
                "rows_skipped_duplicate_conversation": rows_skipped_duplicate_conversation,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp_path, path)


def _rows_checkpoint_exact_path(tries_dir: str, limit: int) -> str:
    """Path for ``rows_{limit:08d}.pkl`` (same naming as checkpoint writer)."""
    return os.path.join(tries_dir, f"rows_{limit:08d}.pkl")


def _try_load_conversation_aggregates_from_rows_checkpoints(
    tries_dir: str,
    *,
    limit: int,
) -> (
    tuple[
        dict[str, int],
        dict[str, int],
        str,
        bool,
        int | None,
        int | None,
        int,
        int,
    ]
    | None
):
    """Load conversation aggregates from ``rows_*.pkl`` (unpickle is heavy).

    Tries the canonical ``rows_{limit:08d}.pkl`` first, then up to
    ``ROWS_CHECKPOINT_MAX_LOAD_ATTEMPTS - 1`` other files ordered by **file size**
    (largest first — usually the final full trie snapshot). Within each load, prefers
    ``payload[\"limit\"] == limit``; otherwise records the best **complete** checkpoint
    (``next_row == payload[\"limit\"]``) with the largest ``limit`` (caller warns if
    below current split).

    Returns
    ``(row_counts, token_totals, path, full_run, checkpoint_next_row, payload_limit,
    rows_missing_conversation_key, rows_skipped_duplicate_conversation)`` where
    ``full_run`` means ``checkpoint_next_row == limit`` (current dataset row cap).
    """
    import pickle

    if not os.path.isdir(tries_dir):
        return None

    numbered: list[tuple[int, str]] = []
    for name in os.listdir(tries_dir):
        m = re.fullmatch(r"rows_(\d+)\.pkl", name)
        if m:
            numbered.append((int(m.group(1)), name))
    if not numbered:
        return None

    sized: list[tuple[int, str]] = []
    for _row_end, name in numbered:
        p = os.path.join(tries_dir, name)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        sized.append((sz, p))
    sized.sort(key=lambda x: x[0], reverse=True)

    paths_to_try: list[str] = []
    seen: set[str] = set()
    hardcoded = os.path.join(tries_dir, HARDCODED_ROWS_CHECKPOINT_BASENAME)
    if os.path.isfile(hardcoded):
        paths_to_try.append(hardcoded)
        seen.add(hardcoded)
    else:
        exact = _rows_checkpoint_exact_path(tries_dir, limit)
        if os.path.isfile(exact):
            paths_to_try.append(exact)
            seen.add(exact)
        for sz, p in sized:
            if p in seen:
                continue
            seen.add(p)
            paths_to_try.append(p)
            if len(paths_to_try) >= ROWS_CHECKPOINT_MAX_LOAD_ATTEMPTS:
                break

    best_relaxed: tuple[int, str, dict, dict, int, int, int] | None = None

    for path in paths_to_try:
        base = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as e:
            print(f"[prefix_trie] skip {base}: unpickle error ({type(e).__name__})", flush=True)
            continue
        if not isinstance(payload, dict):
            print(f"[prefix_trie] skip {base}: not a dict payload", flush=True)
            continue

        crc = payload.get("conversation_row_counts")
        ctt = payload.get("conversation_token_totals")
        if not isinstance(crc, dict) or not isinstance(ctt, dict):
            print(
                f"[prefix_trie] skip {base}: missing conversation_row_counts / "
                "conversation_token_totals",
                flush=True,
            )
            continue
        if payload.get("conversation_aggregates_version") != CONVERSATION_AGGREGATES_VERSION:
            print(
                f"[prefix_trie] skip {base}: conversation_aggregates_version "
                f"{payload.get('conversation_aggregates_version')!r} "
                f"!= {CONVERSATION_AGGREGATES_VERSION}",
                flush=True,
            )
            continue
        rm_ck = payload.get("rows_missing_conversation_key")
        rs_ck = payload.get("rows_skipped_duplicate_conversation")
        if not isinstance(rm_ck, int) or not isinstance(rs_ck, int):
            print(
                f"[prefix_trie] skip {base}: missing rows_missing_conversation_key / "
                "rows_skipped_duplicate_conversation",
                flush=True,
            )
            continue

        plimit = payload.get("limit")
        nrow = payload.get("next_row")
        if not isinstance(plimit, int) or not isinstance(nrow, int):
            print(
                f"[prefix_trie] skip {base}: invalid limit/next_row types",
                flush=True,
            )
            continue

        if plimit == limit:
            full_run = nrow == limit
            ckpt_next = nrow
            try:
                sz_gb = os.path.getsize(path) / (1024**3)
                sz_note = f", {sz_gb:.2f} GiB on disk"
            except OSError:
                sz_note = ""
            print(
                f"[prefix_trie] using rows checkpoint {base} (limit={plimit:,}{sz_note})",
                flush=True,
            )
            return crc, ctt, path, full_run, ckpt_next, plimit, rm_ck, rs_ck

        if nrow == plimit:
            if best_relaxed is None or plimit > best_relaxed[0]:
                best_relaxed = (plimit, path, crc, ctt, nrow, rm_ck, rs_ck)
        else:
            print(
                f"[prefix_trie] skip {base}: incomplete (next_row={nrow:,} "
                f"vs checkpoint limit={plimit:,})",
                flush=True,
            )

    if best_relaxed is not None:
        plimit, path, crc, ctt, nrow, rm_ck, rs_ck = best_relaxed
        base = os.path.basename(path)
        full_run = nrow == limit
        try:
            sz_gb = os.path.getsize(path) / (1024**3)
            sz_note = f", {sz_gb:.2f} GiB on disk"
        except OSError:
            sz_note = ""
        print(
            f"[prefix_trie] using rows checkpoint {base} (checkpoint limit={plimit:,}, "
            f"current dataset limit={limit:,}; stats cover the checkpoint run only"
            f"{sz_note})",
            flush=True,
        )
        return crc, ctt, path, full_run, nrow, plimit, rm_ck, rs_ck

    return None


def _resolve_conversation_aggregates_stats_only(
    root: str,
    *,
    limit: int,
    n_total: int,
    min_sequence_len: int,
    conv_col: str,
    skip_rows_checkpoints: bool = False,
) -> (
    tuple[
        dict[str, int],
        dict[str, int],
        str,
        bool,
        int | None,
        int | None,
        int,
        int,
    ]
    | None
):
    """Return (row_counts, token_totals, source, full_run, ckpt_next, payload_limit,
    rows_missing_conversation_key, rows_skipped_duplicate_conversation).

    When ``skip_rows_checkpoints`` (``--conversation-stats-only``), only tries small
    aggregate pickles — avoids unpickling multi-GB ``rows_*.pkl`` that often lack
    ``conversation_*`` keys on older runs; caller may scan the dataset instead.
    """

    tries_dir = os.path.join(root, "prefix_trie_checkpoints")

    if not skip_rows_checkpoints:
        from_ckpt = _try_load_conversation_aggregates_from_rows_checkpoints(tries_dir, limit=limit)
        if from_ckpt is not None:
            crc, ctt, ck_path, full_run, ckpt_next, plimit, rm_ck, rs_ck = from_ckpt
            return (
                crc,
                ctt,
                f"rows checkpoint ({ck_path!r})",
                full_run,
                ckpt_next,
                plimit,
                rm_ck,
                rs_ck,
            )

    for path, label in (
        (_conversation_aggregates_path(root), "dataset root aggregate file"),
        (
            _conversation_aggregates_checkpoint_path(root),
            "prefix_trie_checkpoints/conversation_aggregates.pkl",
        ),
    ):
        got = _try_load_conversation_aggregates(
            path,
            limit=limit,
            n_total=n_total,
            min_sequence_len=min_sequence_len,
            conv_col=conv_col,
        )
        if got is not None:
            crc, ctt, rm, rs = got
            return crc, ctt, f"{label} ({path!r})", True, None, limit, rm, rs

    return None


def _emit_conversation_stats(
    conv_col: str | None,
    conv_stats_incomplete: bool,
    conv_row_counts: dict[str, int],
    conv_token_totals: dict[str, int],
    *,
    tokens_per_conversation_semantic: str = "longest_row_by_message_count_first_matching_message_column",
    tokens_per_conversation_caption: str = (
        "Tokens / conversation (tiktoken gpt-4o, longest row per key by message count):  "
    ),
    survivor_token_lengths: list[int] | None = None,
) -> dict | None:
    """Print the conversation block; return the JSON-able ``conversation_stats`` dict.

    If ``survivor_token_lengths`` is set (content-hash dedup runs), the token distribution
    is over **those rows only** (one length per maximal prefix chain), not one value per
    conversation key with zeros for non-survivors.
    """
    if conv_col and not conv_stats_incomplete and conv_row_counts:
        import time

        t_conv = time.time()
        n_distinct = len(conv_row_counts)
        print(
            f"\n[stats] conversation-level metrics: {n_distinct:,} distinct keys | "
            "counts, duplicates, distributions...",
            flush=True,
        )
        dup_conv_keys = sum(1 for c in conv_row_counts.values() if c > 1)
        extra_dup_rows = sum(c - 1 for c in conv_row_counts.values() if c > 1)
        rows_attributed = sum(conv_row_counts.values())
        print(
            f"  [stats] building rows-per-conversation and tokens-per-conversation vectors "
            f"({n_distinct:,} entries)...",
            flush=True,
        )
        rows_per_conv = list(conv_row_counts.values())
        if survivor_token_lengths is not None:
            tok_per_conv = list(survivor_token_lengths)
            tok_sem = "tiktoken_gpt4o_per_content_hash_survivor_row"
            tok_caption = (
                "Tokens per prefix-survivor row (tiktoken gpt-4o, after content-hash dedup):  "
            )
            tok_log = "tokens per prefix-survivor row"
        else:
            tok_per_conv = [conv_token_totals.get(k, 0) for k in conv_row_counts]
            tok_sem = tokens_per_conversation_semantic
            tok_caption = tokens_per_conversation_caption
            tok_log = "tokens per conversation"
        print(f"  [stats] summarizing row-count distribution...", flush=True)
        row_dist = _summarize_int_distribution(rows_per_conv, log_label="rows per conversation")
        print(f"  [stats] summarizing token-count distribution...", flush=True)
        tok_dist = _summarize_int_distribution(tok_per_conv, log_label=tok_log)
        print(
            f"  [stats] ranking conversations for top-10 display ({n_distinct:,} keys)...",
            flush=True,
        )
        top_by_rows = sorted(((c, k) for k, c in conv_row_counts.items()), reverse=True)[:10]

        print(f"\n--- Conversation stats (column {conv_col!r}) ---")
        print(f"Rows with non-empty {conv_col}:     {rows_attributed:>14,}")
        print(f"Distinct conversations:             {n_distinct:>14,}")
        print(f"Conversations with >1 row:          {dup_conv_keys:>14,}")
        print(f"Extra duplicate rows Σ(count−1):    {extra_dup_rows:>14,}")
        print(
            f"Rows / conversation:  min={row_dist['min']:,}  max={row_dist['max']:,}  "
            f"mean={row_dist['mean']:.2f}  stdev={row_dist['stdev']:.2f}  "
            f"p50={row_dist['p50']:,}  p90={row_dist['p90']:,}  p99={row_dist['p99']:,}"
        )
        print(
            f"{tok_caption}"
            f"min={tok_dist['min']:,}  max={tok_dist['max']:,}  "
            f"mean={tok_dist['mean']:.2f}  stdev={tok_dist['stdev']:.2f}  "
            f"p50={tok_dist['p50']:,}  p90={tok_dist['p90']:,}  p99={tok_dist['p99']:,}"
        )
        if survivor_token_lengths is not None:
            print(
                f"  (token stats over {len(survivor_token_lengths):,} survivor rows, "
                f"not {n_distinct:,} conversation keys)",
                flush=True,
            )
        print("Top 10 conversations by row count:")
        for cnt, key in top_by_rows:
            print(f"  {cnt:>10,}  rows  {key!s}")

        print(
            f"  [stats] conversation block finished in {time.time() - t_conv:,.1f}s",
            flush=True,
        )

        out = {
            "conversation_key_column": conv_col,
            "tokens_per_conversation_semantic": tok_sem,
            "rows_with_conversation_key": rows_attributed,
            "distinct_conversations": n_distinct,
            "conversations_with_duplicate_rows": dup_conv_keys,
            "extra_duplicate_rows": extra_dup_rows,
            "rows_per_conversation": row_dist,
            "tokens_per_conversation": tok_dist,
            "top_conversations_by_row_count": [
                {"conversation_key": k, "row_count": c} for c, k in top_by_rows
            ],
        }
        if survivor_token_lengths is not None:
            out["token_distribution_survivor_row_count"] = len(survivor_token_lengths)
            out["token_distribution_scope"] = "content_hash_survivor_rows_only"
        return out
    if conv_col and conv_stats_incomplete:
        print(
            "\n--- Conversation stats ---\n"
            "Skipped (resume checkpoint without conversation aggregates). "
            "Delete prefix_trie_checkpoints/ and rerun from row 0 for these metrics."
        )
        return {
            "skipped": True,
            "reason": "checkpoint_missing_conversation_aggregates",
            "conversation_key_column": conv_col,
        }
    if conv_col and not conv_stats_incomplete and not conv_row_counts:
        print(
            f"\n--- Conversation stats (column {conv_col!r}) ---\n"
            "No rows had a non-empty conversation key in the processed range."
        )
        return {
            "skipped": True,
            "reason": "no_non_empty_conversation_keys_in_processed_rows",
            "conversation_key_column": conv_col,
        }
    if not conv_col:
        return {"skipped": True, "reason": "no_conversation_column"}
    return None


def _summarize_int_distribution(
    values: list[int],
    *,
    log_label: str = "",
) -> dict:
    """Mean, stdev, min, max, and percentiles for a non-empty list of ints."""
    import statistics
    import time

    if not values:
        return {
            "n": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    n = len(values)
    if n >= DISTRIBUTION_SORT_LOG_MIN:
        t0 = time.time()
        suffix = f" ({log_label})" if log_label else ""
        print(
            f"  [stats] sorting {n:,} values{suffix} for percentiles...",
            flush=True,
        )
    vals = sorted(values)
    if n >= DISTRIBUTION_SORT_LOG_MIN:
        suffix = f" ({log_label})" if log_label else ""
        print(
            f"  [stats] sort done{suffix} in {time.time() - t0:,.1f}s",
            flush=True,
        )

    def pct(p: float) -> int:
        if n == 1:
            return vals[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[idx]

    return {
        "n": n,
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if n > 1 else 0.0,
        "min": vals[0],
        "max": vals[-1],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
    }


def _user_key(
    row: dict,
    columns: set[str],
    row_index: int,
    user_key_column: str | None,
) -> str:
    if user_key_column:
        v = row.get(user_key_column)
        if v is not None and str(v) != "":
            return str(v)
        return f"missing:{user_key_column}:{row_index}"
    for k in USER_KEY_CANDIDATES:
        if k in columns:
            v = row.get(k)
            if v is not None and str(v) != "":
                return str(v)
    return f"row:{row_index}"


def _row_to_token_ids(
    row: dict,
    enc,
    *,
    message_source: str | None = None,
) -> list[int]:
    messages = _messages_from_row(row, message_source=message_source)
    out: list[int] = []
    for msg in messages:
        if isinstance(msg, dict):
            text = _content_to_str(msg.get("content"))
        elif isinstance(msg, str):
            text = msg
        else:
            text = ""
        if text:
            out.extend(enc.encode(text, disallowed_special=()))
    return out


def _message_count_from_row(
    row: dict,
    *,
    message_source: str | None = None,
) -> int:
    return len(_messages_from_row(row, message_source=message_source))


class _ContentHashTrieNode:
    __slots__ = ("children", "end_rows", "subtree_max_term_depth")

    def __init__(self) -> None:
        self.children: dict[bytes, _ContentHashTrieNode] = {}
        self.end_rows: list[int] = []
        self.subtree_max_term_depth: int = -1


def _content_sha256_digest_sequence(
    row: dict,
    *,
    message_source: str | None,
) -> tuple[bytes, ...]:
    """SHA256 of UTF-8 ``content`` per message, in order (role ignored)."""
    import hashlib

    digests: list[bytes] = []
    for msg in _messages_from_row(row, message_source=message_source):
        if isinstance(msg, dict):
            text = _content_to_str(msg.get("content"))
        elif isinstance(msg, str):
            text = msg
        else:
            text = ""
        digests.append(hashlib.sha256(text.encode("utf-8")).digest())
    return tuple(digests)


def _content_hash_partition_key(row: dict, partition_column: str) -> str:
    """Stable bucket key for partitioning prefix dedup (e.g. WildChat ``hashed_ip``)."""
    v = row.get(partition_column)
    if v is None or str(v) == "":
        return "__missing__"
    return str(v)


def _winners_from_single_content_hash_root(root: _ContentHashTrieNode) -> set[int]:
    """Maximal row indices for one content-hash prefix trie (one user bucket or global)."""

    def subtree_max_term_depth(node: _ContentHashTrieNode, depth: int) -> int:
        m = depth if node.end_rows else -1
        for ch in node.children.values():
            m = max(m, subtree_max_term_depth(ch, depth + 1))
        node.subtree_max_term_depth = m
        return m

    subtree_max_term_depth(root, 0)
    winners: set[int] = set()

    def collect(node: _ContentHashTrieNode, depth: int) -> None:
        if node.end_rows and node.subtree_max_term_depth <= depth:
            winners.add(min(node.end_rows))
        for ch in node.children.values():
            collect(ch, depth + 1)

    collect(root, 0)
    return winners


def _maximal_row_indices_content_prefix_sha256(
    dset,
    *,
    limit: int,
    row_batch_size: int,
    message_source: str | None,
    partition_column: str | None = None,
) -> frozenset[int]:
    """Row indices to ingest: maximal under prefix order on per-turn content SHA256 lists.

    Rows ``a`` and ``b`` compare as prefixes iff digests match element-wise up to
    ``min(len(a), len(b))``; equivalently one digest sequence is a prefix of the other.
    Among a chain, only the **longest** sequence is kept (tie at same node: smallest
    row index).

    If ``partition_column`` is set (e.g. ``hashed_ip``), each row is inserted only into
    that column's bucket's trie—prefix dominance is **within bucket**, not global.
    """
    import time

    roots: dict[str, _ContentHashTrieNode] = {}
    t0 = time.time()
    for start in range(0, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = dset[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j
            digests = _content_sha256_digest_sequence(row, message_source=message_source)
            if partition_column:
                pkey = _content_hash_partition_key(row, partition_column)
                root = roots.setdefault(pkey, _ContentHashTrieNode())
            else:
                root = roots.setdefault("__global__", _ContentHashTrieNode())
            node = root
            for d in digests:
                node = node.children.setdefault(d, _ContentHashTrieNode())
            node.end_rows.append(row_index)
        print(
            f"  [content-hash prefix trie] rows {end:,}/{limit:,} | "
            f"{len(roots):,} partition(s) | {time.time() - t0:,.0f}s",
            flush=True,
        )

    winners: set[int] = set()
    for root in roots.values():
        winners |= _winners_from_single_content_hash_root(root)
    print(
        f"  [content-hash prefix trie] maximal rows for radix ingest: {len(winners):,} / {limit:,}",
        flush=True,
    )
    return frozenset(winners)


DEDUP_TRIE_CONVERSATION_KEY = "conversation_key"
DEDUP_TRIE_CONTENT_PREFIX_SHA256 = "content_prefix_sha256"
# Every row ingested as-is (matches ``build_prefix_trie.py`` semantics).
DEDUP_TRIE_INGEST_ALL = "ingest_all_rows"


def _conversation_winners_from_dataset(
    dset,
    *,
    conv_col: str,
    limit: int,
    row_batch_size: int,
    message_source: str | None = None,
    enc=None,
) -> dict[str, int]:
    """Map conversation key -> row index of the longest variant (tie: smallest index).

    When ``enc`` is set (``conversation`` column datasets), compares
    ``len(_row_to_token_ids(..., message_source=message_source))``. Otherwise uses
    message count only (HF sets without a dedicated ``conversation`` column).
    """
    import time

    winners: dict[str, int] = {}
    best_score: dict[str, int] = {}
    use_tokens = enc is not None
    t0 = time.time()
    for start in range(0, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = dset[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j
            ckey = _conversation_key_value(row, conv_col)
            if ckey is None:
                continue
            if use_tokens:
                score = len(_row_to_token_ids(row, enc, message_source=message_source))
            else:
                score = _message_count_from_row(row, message_source=message_source)
            prev = best_score.get(ckey)
            if prev is None or score > prev:
                best_score[ckey] = score
                winners[ckey] = row_index
        elapsed = time.time() - t0
        mode = "tiktoken length" if use_tokens else "message count"
        print(
            f"  [conversation winners] rows {end:,}/{limit:,} | "
            f"{len(winners):,} distinct keys | score={mode} | {elapsed:,.0f}s",
            flush=True,
        )
    return winners


def _scan_dataset_for_conversation_aggregates(
    dset,
    *,
    conv_col: str,
    limit: int,
    row_batch_size: int,
    message_source: str | None = None,
) -> tuple[dict[str, int], dict[str, int], int, int]:
    """Rebuild ``conversation_row_counts`` / ``conversation_token_totals`` by scanning rows.

    Used when ``--conversation-stats-only`` but checkpoints predate those fields.
    Row counts include every row with a non-empty key. Token totals use the winning row
    per key only, same tie-breaking as the trie ingest pass.

    Also returns ``rows_missing_conversation_key`` and ``rows_skipped_duplicate_conversation``
    over ``0..limit`` (same semantics as the trie ingest pass).
    """
    import time

    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    win_enc = enc if message_source == "conversation" else None
    print(
        "[prefix_trie] Resolving winning row per conversation key ("
        f"{'tiktoken length on conversation' if win_enc else 'message count'})...",
        flush=True,
    )
    winners = _conversation_winners_from_dataset(
        dset,
        conv_col=conv_col,
        limit=limit,
        row_batch_size=row_batch_size,
        message_source=message_source,
        enc=win_enc,
    )
    conv_row_counts: dict[str, int] = {}
    conv_token_totals: dict[str, int] = {}
    rows_missing_conversation_key = 0
    rows_skipped_duplicate_conversation = 0
    t0 = time.time()
    for start in range(0, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = dset[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j
            ckey = _conversation_key_value(row, conv_col)
            if ckey is None:
                rows_missing_conversation_key += 1
                continue
            conv_row_counts[ckey] = conv_row_counts.get(ckey, 0) + 1
            if row_index != winners.get(ckey):
                rows_skipped_duplicate_conversation += 1
                continue
            token_ids = _row_to_token_ids(row, enc, message_source=message_source)
            conv_token_totals[ckey] = len(token_ids)
        elapsed = time.time() - t0
        print(
            f"  [conversation scan] rows {end:,}/{limit:,} | "
            f"{len(conv_row_counts):,} distinct conversations | {elapsed:,.0f}s",
            flush=True,
        )
    return (
        conv_row_counts,
        conv_token_totals,
        rows_missing_conversation_key,
        rows_skipped_duplicate_conversation,
    )


def _commit_for_root(root: str) -> None:
    if root.startswith("/lmsys"):
        lmsys_chat_1m_volume.commit()
    elif root.startswith("/datasets"):
        hf_datasets_volume.commit()


@app.function(
    image=image,
    memory=1024 * 256,
    timeout=86400,
    volumes={
        "/lmsys": lmsys_chat_1m_volume,
        "/datasets": hf_datasets_volume,
    },
)
def build_hf_disk_prefix_tries(
    *,
    dataset_disk_path: str | None = None,
    disk_candidates: list[tuple[str, str]] | None = None,
    run_preset: str | None = None,
    extra_candidates_logged: str | None = None,
    user_key_column: str | None = None,
    row_batch_size: int = 512,
    checkpoint_every_rows: int = 250_000,
    min_sequence_len: int = 1,
    max_rows: int | None = None,
    stats_tag: str | None = None,
    conversation_stats_only: bool = False,
    dedup_content_prefix_sha256: bool | None = None,
    ingest_all_rows: bool = False,
):
    import json
    import pickle
    import time

    import tiktoken
    from datasets import Dataset, DatasetDict, load_from_disk

    if dataset_disk_path:
        root = _normalize_hf_disk_root(dataset_disk_path)
    else:
        cands = (
            disk_candidates
            if disk_candidates is not None
            else (LMSYS_DISK_CANDIDATES + WILDCHAT_DISK_CANDIDATES)
        )
        _, root = _find_disk_root(cands)

    if ingest_all_rows:
        if dedup_content_prefix_sha256 is True:
            raise ValueError(
                "ingest_all_rows=True is incompatible with dedup_content_prefix_sha256=True"
            )
        dedup_content_prefix_sha256 = False
        print(
            "[prefix_trie] --ingest-all-rows: every row ingested as-is "
            "(no content-hash prefix dedup, no conversation-winner selection). "
            "Matches build_prefix_trie.py semantics for a directly comparable A/B/C.",
            flush=True,
        )
    elif dedup_content_prefix_sha256 is None:
        dedup_content_prefix_sha256 = _default_dedup_content_prefix_sha256(root, run_preset)
        if dedup_content_prefix_sha256:
            print(
                "[prefix_trie] Default: content-hash prefix dedup "
                "(WildChat disk path or --preset wildchat).",
                flush=True,
            )

    sys.setrecursionlimit(1_000_000)

    dsd = load_from_disk(root)
    if isinstance(dsd, DatasetDict):
        if "train" in dsd:
            dset = dsd["train"]
        else:
            key = next(iter(dsd))
            dset = dsd[key]
    elif isinstance(dsd, Dataset):
        dset = dsd
    else:
        raise RuntimeError(f"Unexpected load_from_disk result: {type(dsd)!r}")

    n_total = len(dset)
    limit = n_total if max_rows is None else min(n_total, max_rows)
    columns = set(dset.column_names)
    conv_col = _conversation_key_column(columns)
    message_source = _dedup_message_source(columns)
    content_hash_partition_column = "hashed_ip" if "hashed_ip" in columns else None
    expected_ch_partition = content_hash_partition_column if dedup_content_prefix_sha256 else None
    if message_source:
        print(
            f"[prefix_trie] Message list for trie + per-key dedup: {message_source!r} "
            "(not messages/conversations fallback for those steps)."
        )
    print(
        f"Loaded {root!r} split len={n_total:,} (processing {limit:,}); "
        f"columns={sorted(columns)[:24]}{'…' if len(columns) > 24 else ''}"
    )
    if conv_col:
        print(f"[prefix_trie] Per-conversation stats use column {conv_col!r}")
    else:
        print(
            "[prefix_trie] No conversation_id / conversation_hash column; "
            "skipping duplicate-conversation and per-conversation token stats."
        )

    tag = stats_tag if stats_tag else _stats_basename(root)

    if conversation_stats_only:
        if not conv_col:
            print(
                "[prefix_trie] --conversation-stats-only needs conversation_id or "
                "conversation_hash on this split."
            )
            return {}
        resolved = _resolve_conversation_aggregates_stats_only(
            root,
            limit=limit,
            n_total=n_total,
            min_sequence_len=min_sequence_len,
            conv_col=conv_col,
            skip_rows_checkpoints=True,
        )
        if resolved is None:
            tries_dir = os.path.join(root, "prefix_trie_checkpoints")
            print(
                f"[prefix_trie] No small aggregate files "
                f"({CONVERSATION_AGGREGATES_FILENAME!r} or "
                f"prefix_trie_checkpoints/{CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME!r}). "
                f"Large rows_*.pkl under {tries_dir!r} are skipped in stats-only mode "
                f"(older checkpoints often lack conversation_* fields). "
                f"Scanning {limit:,} rows with tiktoken…"
            )
            (
                conv_row_counts,
                conv_token_totals,
                rows_missing_conversation_key,
                rows_skipped_duplicate_conversation,
            ) = _scan_dataset_for_conversation_aggregates(
                dset,
                conv_col=conv_col,
                limit=limit,
                row_batch_size=row_batch_size,
                message_source=message_source,
            )
            agg_source = (
                "dataset scan (recomputed; no small aggregate cache; "
                "rows checkpoints not loaded in --conversation-stats-only)"
            )
            aggregates_full_run = True
            ckpt_next = limit
            checkpoint_payload_limit = limit
        else:
            (
                conv_row_counts,
                conv_token_totals,
                agg_source,
                aggregates_full_run,
                ckpt_next,
                checkpoint_payload_limit,
                rows_missing_conversation_key,
                rows_skipped_duplicate_conversation,
            ) = resolved
            print(
                f"[prefix_trie] Loaded conversation aggregates from {agg_source} "
                "(skipping trie build and dataset scan)."
            )
        if checkpoint_payload_limit is not None and checkpoint_payload_limit != limit:
            print(
                f"[prefix_trie] WARNING: aggregates are for checkpoint limit "
                f"{checkpoint_payload_limit:,}; current split limit is {limit:,}."
            )
        elif not aggregates_full_run and ckpt_next is not None:
            print(
                f"[prefix_trie] WARNING: partial checkpoint (next_row={ckpt_next:,} "
                f"vs limit={limit:,}); conversation stats are only for rows processed "
                "before that checkpoint."
            )
        if (
            aggregates_full_run
            and checkpoint_payload_limit == limit
            and ("rows checkpoint" in agg_source or "dataset scan" in agg_source)
        ):
            os.makedirs(os.path.join(root, "prefix_trie_checkpoints"), exist_ok=True)
            for save_path in (
                _conversation_aggregates_path(root),
                _conversation_aggregates_checkpoint_path(root),
            ):
                _save_conversation_aggregates(
                    save_path,
                    limit=limit,
                    n_total=n_total,
                    min_sequence_len=min_sequence_len,
                    conv_col=conv_col,
                    conversation_row_counts=conv_row_counts,
                    conversation_token_totals=conv_token_totals,
                    rows_missing_conversation_key=rows_missing_conversation_key,
                    rows_skipped_duplicate_conversation=rows_skipped_duplicate_conversation,
                    completed_next_row=limit,
                )
            print(
                "[prefix_trie] Wrote small aggregate caches "
                f"({CONVERSATION_AGGREGATES_FILENAME!r} and "
                f"prefix_trie_checkpoints/{CHECKPOINT_CONVERSATION_AGGREGATES_FILENAME!r}) "
                "for faster loads next time."
            )
        print(
            f"\n[ingest accounting] rows_processed={limit:,}  "
            f"rows_missing_conversation_key={rows_missing_conversation_key:,}  "
            f"rows_skipped_duplicate_conversation={rows_skipped_duplicate_conversation:,}  "
            f"(column {conv_col!r})",
            flush=True,
        )
        _t_sem, _t_cap = _dedup_stats_labels(message_source)
        conversation_stats = _emit_conversation_stats(
            conv_col,
            False,
            conv_row_counts,
            conv_token_totals,
            tokens_per_conversation_semantic=_t_sem,
            tokens_per_conversation_caption=_t_cap,
        )
        stats = {
            "dataset_disk_path": root,
            "stats_tag": tag,
            "preset": run_preset,
            "extra_candidates": extra_candidates_logged,
            "dedup_message_source": message_source,
            "content_hash_dedup_partition_column": expected_ch_partition,
            "conversation_stats_only": True,
            "rows_processed": limit,
            "sequences_ingested": None,
            "rows_missing_conversation_key": rows_missing_conversation_key,
            "rows_skipped_duplicate_conversation": rows_skipped_duplicate_conversation,
            "conversation_aggregates_source": agg_source,
            "conversation_aggregates_full_run": aggregates_full_run,
            "checkpoint_next_row": ckpt_next,
            "checkpoint_payload_limit": checkpoint_payload_limit,
            "conversation_stats": conversation_stats,
        }
        stats_path = os.path.join(root, f"prefix_trie_stats_{tag}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        if _is_hf_save_dir(root):
            _commit_for_root(root)
        print(f"\nStats written to {stats_path}")
        return stats

    enc = tiktoken.encoding_for_model("gpt-4o")
    tries_dir = os.path.join(root, "prefix_trie_checkpoints")
    os.makedirs(tries_dir, exist_ok=True)

    global_trie = RadixTrie()
    per_user_tries: dict[str, RadixTrie] = {}
    total_tokens = 0
    total_sequences = 0
    skipped_empty = 0
    resume_row = 0
    conv_row_counts: dict[str, int] = {}
    conv_token_totals: dict[str, int] = {}
    conv_stats_incomplete = False
    rows_missing_conversation_key = 0
    rows_skipped_duplicate_conversation = 0
    prefix_survivor_token_lens: list[int] = []
    prefix_survivor_token_lens_resume_incomplete = False
    if ingest_all_rows:
        expected_dedup_trie = DEDUP_TRIE_INGEST_ALL
    elif dedup_content_prefix_sha256:
        expected_dedup_trie = DEDUP_TRIE_CONTENT_PREFIX_SHA256
    else:
        expected_dedup_trie = DEDUP_TRIE_CONVERSATION_KEY

    ckpt_files = sorted(
        f for f in os.listdir(tries_dir) if f.startswith("rows_") and f.endswith(".pkl")
    )
    if ckpt_files:
        latest = os.path.join(tries_dir, ckpt_files[-1])
        print(f"Resuming from checkpoint: {latest}")
        with open(latest, "rb") as f:
            payload = pickle.load(f)
        ckpt_schema = int(payload.get("trie_build_schema", 1))
        saved_dedup = payload.get("dhfpt_dedup", DEDUP_TRIE_CONVERSATION_KEY)
        saved_ch_part = payload.get("dhfpt_content_hash_partition")
        dedup_mismatch = (
            saved_dedup != expected_dedup_trie or saved_ch_part != expected_ch_partition
        )
        if ckpt_schema < TRIE_CHECKPOINT_SCHEMA_VERSION or dedup_mismatch:
            if dedup_mismatch and ckpt_schema >= TRIE_CHECKPOINT_SCHEMA_VERSION:
                print(
                    "  WARNING: checkpoint trie dedup "
                    f"(mode {saved_dedup!r} vs {expected_dedup_trie!r}, "
                    f"content-hash partition {saved_ch_part!r} vs {expected_ch_partition!r}); "
                    "rebuilding tries from row 0.",
                    flush=True,
                )
            if ckpt_schema < TRIE_CHECKPOINT_SCHEMA_VERSION:
                print(
                    "  WARNING: checkpoint predates current trie schema "
                    f"(trie_build_schema={ckpt_schema} < {TRIE_CHECKPOINT_SCHEMA_VERSION}); "
                    "rebuilding tries from row 0. Delete stale rows_*.pkl to silence this."
                )
            global_trie = RadixTrie()
            per_user_tries = {}
            total_tokens = 0
            total_sequences = 0
            skipped_empty = 0
            resume_row = 0
            conv_row_counts = {}
            conv_token_totals = {}
            rows_missing_conversation_key = 0
            rows_skipped_duplicate_conversation = 0
            prefix_survivor_token_lens = []
            prefix_survivor_token_lens_resume_incomplete = False
        else:
            global_trie = payload["global_trie"]
            per_user_tries = payload["per_user_tries"]
            total_tokens = payload["total_tokens"]
            total_sequences = payload["total_sequences"]
            skipped_empty = payload["skipped_empty"]
            resume_row = payload["next_row"]
            rm_ck = payload.get("rows_missing_conversation_key")
            rs_ck = payload.get("rows_skipped_duplicate_conversation")
            if isinstance(rm_ck, int) and isinstance(rs_ck, int):
                rows_missing_conversation_key = rm_ck
                rows_skipped_duplicate_conversation = rs_ck
            if conv_col and payload.get("conversation_row_counts") is not None:
                conv_row_counts = payload["conversation_row_counts"]
                conv_token_totals = payload.get("conversation_token_totals") or {}
            elif conv_col:
                conv_stats_incomplete = True
                print(
                    "  WARNING: checkpoint has no conversation_row_counts; "
                    "duplicate / per-conversation token stats omitted this run."
                )
            if dedup_content_prefix_sha256:
                ptl = payload.get("prefix_survivor_token_lens")
                if isinstance(ptl, list):
                    prefix_survivor_token_lens = [int(x) for x in ptl]
                elif resume_row > 0:
                    prefix_survivor_token_lens = []
                    prefix_survivor_token_lens_resume_incomplete = True
                    print(
                        "  WARNING: checkpoint has no prefix_survivor_token_lens; "
                        "token distribution falls back to one value per conversation key. "
                        "Remove prefix_trie_checkpoints/ for survivor-only token stats.",
                        flush=True,
                    )
                else:
                    prefix_survivor_token_lens = []
        if payload.get("limit") != limit:
            print(
                f"  WARNING: checkpoint limit={payload.get('limit')} vs current {limit}; "
                "delete prefix_trie_checkpoints/ for a clean run."
            )
        print(
            f"  restored at row {resume_row:,} | {total_sequences:,} seqs, "
            f"{total_tokens:,} toks, {len(per_user_tries):,} user keys"
        )

    conversation_winners: dict[str, int] | None = None
    prefix_ingest_rows: frozenset[int] | None = None
    if ingest_all_rows:
        print(
            "[prefix_trie] Trie dedup: none (ingest every row). "
            "T sums lengths of all rows; intra-user and cross-user savings are "
            "directly comparable to build_prefix_trie.py on the same data.",
            flush=True,
        )
    elif dedup_content_prefix_sha256:
        if expected_ch_partition:
            print(
                f"[prefix_trie] Trie dedup: SHA256(content) per message; maximal prefixes "
                f"within each {expected_ch_partition!r} bucket (same-user chains only).",
                flush=True,
            )
        else:
            print(
                "[prefix_trie] Trie dedup: SHA256(content) per message in order; "
                "keep maximal rows under global prefix order (no hashed_ip column).",
                flush=True,
            )
        prefix_ingest_rows = _maximal_row_indices_content_prefix_sha256(
            dset,
            limit=limit,
            row_batch_size=row_batch_size,
            message_source=message_source,
            partition_column=expected_ch_partition,
        )
    if conv_col and ingest_all_rows:
        print(
            f"[prefix_trie] Per-conversation token totals use max length across all rows "
            f"for each {conv_col!r} (no winner selection; every row is ingested).",
            flush=True,
        )
    elif conv_col and dedup_content_prefix_sha256:
        print(
            f"[prefix_trie] Per-conversation token totals (pickle/cache) use max length among "
            f"prefix-ingested rows ({conv_col!r}). Printed token *distribution* is one value "
            "per content-hash survivor row (longest chains after dedup). Row counts still "
            "include every row with a key.",
            flush=True,
        )
    elif conv_col:
        win_enc = enc if message_source == "conversation" else None
        score_desc = (
            "tiktoken gpt-4o length on column 'conversation'"
            if win_enc
            else "message count (first matching messages/conversations column)"
        )
        print(
            f"[prefix_trie] Longest row per {conv_col!r} ({score_desc}; tie: earliest "
            "row index). Only those rows are inserted into tries.",
            flush=True,
        )
        conversation_winners = _conversation_winners_from_dataset(
            dset,
            conv_col=conv_col,
            limit=limit,
            row_batch_size=row_batch_size,
            message_source=message_source,
            enc=win_enc,
        )

    t0 = time.time()
    warned_user_key = False

    for start in range(resume_row, limit, row_batch_size):
        end = min(start + row_batch_size, limit)
        batch = dset[start:end]
        keys = list(batch.keys())
        batch_len = len(batch[keys[0]]) if keys else 0
        for j in range(batch_len):
            row = {k: batch[k][j] for k in keys}
            row_index = start + j

            ckey = _conversation_key_value(row, conv_col)
            if conv_col and ckey is None:
                rows_missing_conversation_key += 1
            if dedup_content_prefix_sha256 and prefix_ingest_rows is not None:
                if row_index not in prefix_ingest_rows:
                    rows_skipped_duplicate_conversation += 1
            elif conv_col and ckey is not None:
                if conversation_winners is not None and row_index != conversation_winners.get(ckey):
                    rows_skipped_duplicate_conversation += 1
            if ckey is not None and not conv_stats_incomplete:
                conv_row_counts[ckey] = conv_row_counts.get(ckey, 0) + 1

            uid = _user_key(row, columns, row_index, user_key_column)
            if uid.startswith("row:") and not warned_user_key and not user_key_column:
                warned_user_key = True
                print(
                    "[warn] No user-id column found; using row index as key — "
                    "intra-user overlap (A) will be ~0. Set --user-key-column if "
                    "your schema has a grouping column."
                )

            if dedup_content_prefix_sha256:
                ingest_trie = prefix_ingest_rows is not None and row_index in prefix_ingest_rows
                need_tokenize = ingest_trie
            else:
                ingest_trie = True
                need_tokenize = (
                    conv_col is None
                    or ckey is None
                    or conversation_winners is None
                    or row_index == conversation_winners.get(ckey)
                )
                if (
                    conversation_winners is not None
                    and ckey is not None
                    and row_index != conversation_winners.get(ckey)
                ):
                    ingest_trie = False

            if need_tokenize:
                token_ids = _row_to_token_ids(row, enc, message_source=message_source)
            else:
                token_ids = []
            if ckey is not None and not conv_stats_incomplete:
                if dedup_content_prefix_sha256:
                    if prefix_ingest_rows is not None and row_index in prefix_ingest_rows:
                        conv_token_totals[ckey] = max(
                            conv_token_totals.get(ckey, 0),
                            len(token_ids),
                        )
                elif ingest_all_rows:
                    conv_token_totals[ckey] = max(
                        conv_token_totals.get(ckey, 0),
                        len(token_ids),
                    )
                elif conversation_winners is not None and row_index == conversation_winners.get(
                    ckey
                ):
                    conv_token_totals[ckey] = len(token_ids)

            if (
                dedup_content_prefix_sha256
                and prefix_ingest_rows is not None
                and row_index in prefix_ingest_rows
                and need_tokenize
            ):
                prefix_survivor_token_lens.append(len(token_ids))

            if ingest_trie:
                if len(token_ids) < min_sequence_len:
                    skipped_empty += 1
                    continue
                seq = array("I", token_ids)

                global_trie.insert(seq)
                user_trie = per_user_tries.get(uid)
                if user_trie is None:
                    user_trie = RadixTrie()
                    per_user_tries[uid] = user_trie
                user_trie.insert(seq)

                total_tokens += len(seq)
                total_sequences += 1

        elapsed = time.time() - t0
        print(
            f"  rows {end:,}/{limit:,} | {total_sequences:,} seqs, {total_tokens:,} toks | "
            f"{elapsed:,.0f}s elapsed"
        )

        crossed = (end // checkpoint_every_rows) > (start // checkpoint_every_rows)
        if end == limit or crossed:
            ckpt_path = os.path.join(tries_dir, f"rows_{end:08d}.pkl")
            tmp_path = ckpt_path + ".tmp"
            payload = {
                "trie_build_schema": TRIE_CHECKPOINT_SCHEMA_VERSION,
                "dhfpt_dedup": expected_dedup_trie,
                "dhfpt_content_hash_partition": expected_ch_partition,
                "conversation_aggregates_version": CONVERSATION_AGGREGATES_VERSION,
                "next_row": end,
                "limit": limit,
                "total_sequences": total_sequences,
                "total_tokens": total_tokens,
                "skipped_empty": skipped_empty,
                "rows_missing_conversation_key": rows_missing_conversation_key,
                "rows_skipped_duplicate_conversation": rows_skipped_duplicate_conversation,
                "global_trie": global_trie,
                "per_user_tries": per_user_tries,
                "conversation_row_counts": conv_row_counts,
                "conversation_token_totals": conv_token_totals,
            }
            if dedup_content_prefix_sha256:
                payload["prefix_survivor_token_lens"] = list(prefix_survivor_token_lens)
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, ckpt_path)
            if end == limit and conv_col and conv_row_counts and not conv_stats_incomplete:
                _save_conversation_aggregates(
                    _conversation_aggregates_checkpoint_path(root),
                    limit=limit,
                    n_total=n_total,
                    min_sequence_len=min_sequence_len,
                    conv_col=conv_col,
                    conversation_row_counts=conv_row_counts,
                    conversation_token_totals=conv_token_totals,
                    rows_missing_conversation_key=rows_missing_conversation_key,
                    rows_skipped_duplicate_conversation=rows_skipped_duplicate_conversation,
                    completed_next_row=end,
                )
            if _is_hf_save_dir(root):
                _commit_for_root(root)
            print(f"    checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:,.1f} MB)")

    if total_sequences == 0:
        print("No sequences ingested; nothing to report.")
        return {}

    if conv_col:
        print(
            f"\n[ingest accounting] rows_processed={limit:,}  "
            f"sequences_ingested={total_sequences:,}  "
            f"rows_missing_conversation_key={rows_missing_conversation_key:,}  "
            f"rows_skipped_duplicate_conversation={rows_skipped_duplicate_conversation:,}  "
            f"(column {conv_col!r})",
            flush=True,
        )
        att = (
            sum(conv_row_counts.values()) if conv_row_counts and not conv_stats_incomplete else None
        )
        if att is not None and not conv_stats_incomplete:
            check = rows_missing_conversation_key + att
            if check != limit:
                print(
                    f"  [ingest accounting] note: missing + rows_with_key ({check:,}) != "
                    f"limit ({limit:,})",
                    flush=True,
                )
    else:
        print(
            f"\n[ingest accounting] rows_processed={limit:,}  "
            f"sequences_ingested={total_sequences:,}  "
            "(no conversation_id / conversation_hash column)",
            flush=True,
        )

    print("\nComputing overlap stats...", flush=True)
    t_overlap = time.time()
    n_user_keys = len(per_user_tries)
    print(
        f"  [stats] {n_user_keys:,} user/group tries | global trie unique_token_count()...",
        flush=True,
    )
    global_unique = global_trie.unique_token_count()
    print(
        f"  [stats] global U(all)={global_unique:,} ({time.time() - t_overlap:,.1f}s)",
        flush=True,
    )

    sum_intra_unique = 0
    user_count = 0
    top_users = []
    if n_user_keys <= 0:
        log_every = 1
    else:
        log_every = max(1, n_user_keys // 25)
        log_every = min(log_every, OVERLAP_STATS_LOG_EVERY_USERS)
    for uid, user_trie in per_user_tries.items():
        u = user_trie.unique_token_count()
        sum_intra_unique += u
        user_count += 1
        top_users.append((user_trie.total_tokens_inserted, u, uid))
        if user_count % log_every == 0:
            print(
                f"  [stats] per-user unique: {user_count:,}/{n_user_keys:,} keys | "
                f"Σ_u U(R_u) so far={sum_intra_unique:,} | {time.time() - t_overlap:,.1f}s",
                flush=True,
            )

    print(
        f"  [stats] per-user unique counts done ({n_user_keys:,} keys) in "
        f"{time.time() - t_overlap:,.1f}s",
        flush=True,
    )

    intra_savings = total_tokens - sum_intra_unique
    global_savings = total_tokens - global_unique
    cross_user_extra = sum_intra_unique - global_unique

    print(f"\n{'=' * 72}")
    print(f"Dataset root:                    {root}")
    print(f"Rows processed:                {limit:>14,}")
    print(f"Sequences inserted:            {total_sequences:>14,}")
    if conv_col:
        print(f"Rows missing {conv_col}:       {rows_missing_conversation_key:>14,}")
        print(f"Rows skipped (dup conv):       {rows_skipped_duplicate_conversation:>14,}")
    print(f"User / group keys:             {user_count:>14,}")
    print(f"Empty / short skipped:         {skipped_empty:>14,}")
    print(f"Total tokens T:                {total_tokens:>14,}")
    print(f"{'-' * 72}")
    print("(A) INTRA-USER overlap   -- prefixes shared within the same user/group key")
    print(
        f"      (T - sum_u U(R_u)) / T   "
        f"= {intra_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(intra_savings, total_tokens)}"
    )
    print()
    print("(C) CROSS-USER overlap   -- extra sharing from pooling across keys")
    print(
        f"      (sum_u U(R_u) - U(all)) / T   "
        f"= {cross_user_extra:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(cross_user_extra, total_tokens)}"
    )
    print(f"{'-' * 72}")
    print("(B) GLOBAL overlap")
    print(
        f"      (T - U(all)) / T         "
        f"= {global_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(global_savings, total_tokens)}"
    )
    print(
        f"      A + C                     "
        f"= {_fmt_pct(intra_savings + cross_user_extra, total_tokens)}"
    )
    print(f"{'=' * 72}")

    _t_sem, _t_cap = _dedup_stats_labels(message_source)
    survivor_tok_lens = None
    if dedup_content_prefix_sha256 and not prefix_survivor_token_lens_resume_incomplete:
        survivor_tok_lens = prefix_survivor_token_lens
    conversation_stats = _emit_conversation_stats(
        conv_col,
        conv_stats_incomplete,
        conv_row_counts,
        conv_token_totals,
        tokens_per_conversation_semantic=_t_sem,
        tokens_per_conversation_caption=_t_cap,
        survivor_token_lengths=survivor_tok_lens,
    )

    if (
        conv_col
        and not conv_stats_incomplete
        and conv_row_counts
        and conversation_stats
        and not conversation_stats.get("skipped")
    ):
        agg_path = _conversation_aggregates_path(root)
        _save_conversation_aggregates(
            agg_path,
            limit=limit,
            n_total=n_total,
            min_sequence_len=min_sequence_len,
            conv_col=conv_col,
            conversation_row_counts=conv_row_counts,
            conversation_token_totals=conv_token_totals,
            rows_missing_conversation_key=rows_missing_conversation_key,
            rows_skipped_duplicate_conversation=rows_skipped_duplicate_conversation,
            completed_next_row=limit,
        )
        print(f"[prefix_trie] Wrote conversation aggregates cache {agg_path!r}")

    top_users.sort(reverse=True)
    print("\nTop 10 groups by tokens inserted:")
    print(f"  {'tokens':>14}  {'unique':>14}  {'savings':>8}  key")
    for total_t, uniq_t, th in top_users[:10]:
        saved = total_t - uniq_t
        print(f"  {total_t:>14,}  {uniq_t:>14,}  {_fmt_pct(saved, total_t)}  {th!s}")

    stats = {
        "dataset_disk_path": root,
        "stats_tag": tag,
        "preset": run_preset,
        "extra_candidates": extra_candidates_logged,
        "dedup_message_source": message_source,
        "dedup_trie_mode": expected_dedup_trie,
        "content_hash_dedup_partition_column": expected_ch_partition,
        "user_key_column": user_key_column,
        "row_batch_size": row_batch_size,
        "min_sequence_len": min_sequence_len,
        "max_rows": max_rows,
        "rows_processed": limit,
        "total_sequences": total_sequences,
        "sequences_ingested": total_sequences,
        "total_tokens": total_tokens,
        "skipped_empty": skipped_empty,
        "rows_missing_conversation_key": (rows_missing_conversation_key if conv_col else None),
        "rows_skipped_duplicate_conversation": (
            rows_skipped_duplicate_conversation if conv_col else None
        ),
        "user_count": user_count,
        "global_unique_tokens": global_unique,
        "sum_intra_unique_tokens": sum_intra_unique,
        "intra_user_savings": intra_savings,
        "cross_user_extra_savings": cross_user_extra,
        "global_savings": global_savings,
        "intra_user_savings_pct": (
            100.0 * intra_savings / total_tokens if total_tokens > 0 else None
        ),
        "cross_user_extra_pct": (
            100.0 * cross_user_extra / total_tokens if total_tokens > 0 else None
        ),
        "global_savings_pct": (100.0 * global_savings / total_tokens if total_tokens > 0 else None),
        "elapsed_seconds": time.time() - t0,
        "top_groups": [
            {"key": k, "tokens": tt, "unique_tokens": uu} for tt, uu, k in top_users[:10]
        ],
        "conversation_stats": conversation_stats,
    }

    stats_path = os.path.join(root, f"prefix_trie_stats_{tag}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    _commit_for_root(root)
    print(f"\nStats written to {stats_path}")
    return stats


@app.local_entrypoint()
def prefix_trie(
    preset: str = "auto",
    dataset_disk_path: str | None = None,
    extra_candidates: str | None = None,
    user_key_column: str | None = None,
    row_batch_size: int = 512,
    checkpoint_every_rows: int = 250_000,
    min_sequence_len: int = 1,
    max_rows: int | None = None,
    stats_tag: str | None = None,
    conversation_stats_only: bool = False,
    dedup_content_prefix_sha256: bool = False,
    no_dedup_content_prefix_sha256: bool = False,
    ingest_all_rows: bool = False,
):
    """KV prefix-trie overlap stats for HF ``save_to_disk`` data on ``/lmsys`` or ``/datasets`` mounts.

    * ``--preset auto`` — search LMSYS default paths, then WildChat (first with valid metadata wins).
    * ``--preset lmsys`` / ``wildchat`` — only that dataset’s default paths.
    * ``--dataset-disk-path /abs/root`` — use this tree; ``preset`` / ``extra_candidates`` ignored for resolution.
    * ``--extra-candidates /path/a,/path/b`` — try these directories first (comma-separated), then ``preset`` paths.
    * ``--conversation-stats-only`` — small aggregate pickles only, else full split scan
      (tiktoken); does not load ``rows_*.pkl``. Writes aggregate caches when recomputing.
    * Content-hash prefix dedup — radix trie ingests only maximal rows under per-message
      content SHA256 prefix order (see module docstring). **Default on** for WildChat
      (path or ``--preset wildchat``). **Default off** for LMSYS. Pass
      ``--dedup-content-prefix-sha256`` to force on; ``--no-dedup-content-prefix-sha256``
      to force off (e.g. WildChat with legacy per-conversation-key winners).
    * ``--ingest-all-rows`` — disable **both** the content-hash prefix dedup **and** the
      per-conversation winner selection; every row is tokenized and inserted. This is
      the mode to use when you want A/B/C numbers directly comparable to
      ``build_prefix_trie.py`` (same "no dedup, one sequence per row" semantics; still
      differs in tokenizer — tiktoken gpt-4o — and in user/group key). Incompatible
      with ``--dedup-content-prefix-sha256``; implies ``--no-dedup-content-prefix-sha256``.
    """
    if dedup_content_prefix_sha256 and no_dedup_content_prefix_sha256:
        raise ValueError(
            "Use at most one of --dedup-content-prefix-sha256 and --no-dedup-content-prefix-sha256"
        )
    if ingest_all_rows and dedup_content_prefix_sha256:
        raise ValueError("--ingest-all-rows cannot be combined with --dedup-content-prefix-sha256")
    if dedup_content_prefix_sha256:
        dedup_opt: bool | None = True
    elif no_dedup_content_prefix_sha256 or ingest_all_rows:
        dedup_opt = False
    else:
        dedup_opt = None

    disk_candidates: list[tuple[str, str]] | None = None
    if dataset_disk_path is None:
        disk_candidates = _merge_disk_candidates(preset, extra_candidates)

    build_hf_disk_prefix_tries.remote(
        dataset_disk_path=dataset_disk_path,
        disk_candidates=disk_candidates,
        run_preset=preset,
        extra_candidates_logged=extra_candidates,
        user_key_column=user_key_column,
        row_batch_size=row_batch_size,
        checkpoint_every_rows=checkpoint_every_rows,
        min_sequence_len=min_sequence_len,
        max_rows=max_rows,
        stats_tag=stats_tag,
        conversation_stats_only=conversation_stats_only,
        dedup_content_prefix_sha256=dedup_opt,
        ingest_all_rows=ingest_all_rows,
    )
