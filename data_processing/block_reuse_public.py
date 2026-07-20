"""Block-level (256-token) prefix-reuse stats for the public HF datasets.

Mirrors ``data_processing/week_reuse_stats.py`` (which measures ART-Chat-2.5M)
so all three datasets are comparable on the *same* request-row, block-level
metric the KV cache actually keys on: prefix-cumulative SHA-256 over 256-token
blocks, token-weighted, split into global / intra-user / cross-user reuse.

Two phases:

* Phase 1 (highly parallel, CPU fan-out): each worker tokenizes a row range
  (tiktoken ``gpt-4o``, same as ``build_hf_prefix_trie``), computes the block
  keys per request, and writes a compact per-shard pickle to the volume.
* Phase 2 (single reducer): streams the shards in dataset order and counts
  reuse with identical logic to ``week_reuse_stats.block_stats``.

User key matches ``build_hf_prefix_trie`` (``hashed_ip`` for WildChat;
LMSYS has no user identity, so it falls back to ``conversation_id`` and only
the global figure is meaningful). Every row is ingested as-is (no dedup), so
the multi-turn staircase is counted the same way it is for the production trace.

Usage::

    modal run --detach --env=alessio-dev data_processing/block_reuse_public.py::run --preset wildchat
    modal run --detach --env=alessio-dev data_processing/block_reuse_public.py::run --preset lmsys
"""

from __future__ import annotations

import hashlib
import json
from array import array

import modal

from app import app, hf_datasets_volume, lmsys_chat_1m_volume
from build_hf_prefix_trie import (
    _dedup_message_source,
    _find_disk_root,
    _merge_disk_candidates,
    _normalize_hf_disk_root,
    _row_to_token_ids,
    _user_key,
)

image = (
    modal.Image.debian_slim()
    .pip_install("datasets>=3.0", "pyarrow", "tiktoken")
    .add_local_python_source("app", "build_eval_dataset", "build_hf_prefix_trie", "gorgo")
)

BLOCK_SIZE = 256
SHARD_ROWS = 15_000
TMP_ROOT = "/datasets/_block_reuse_tmp"

VOLUMES = {"/datasets": hf_datasets_volume, "/lmsys": lmsys_chat_1m_volume}


def _block_keys(token_ids: list[int], block_size: int) -> array:
    """Prefix-cumulative SHA-256 block hashes as 64-bit keys.

    Identical keying to ``export_metadata_trace._block_digests`` +
    ``week_reuse_stats`` (which used ``int(hexdigest[:12][:16], 16)``, i.e. the
    first 8 bytes of the digest), so reuse is measured on the same key space.
    """
    keys = array("Q")
    prev = b""
    n = len(token_ids)
    for i in range(0, n, block_size):
        block = token_ids[i : i + block_size]
        h = hashlib.sha256()
        h.update(prev)
        h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in block))
        digest = h.digest()
        keys.append(int.from_bytes(digest[:8], "big"))
        prev = digest
    return keys


def _load_split(root: str):
    from datasets import Dataset, DatasetDict, load_from_disk

    dsd = load_from_disk(root)
    if isinstance(dsd, DatasetDict):
        return dsd["train"] if "train" in dsd else dsd[next(iter(dsd))]
    if isinstance(dsd, Dataset):
        return dsd
    raise RuntimeError(f"unexpected load_from_disk result: {type(dsd)!r}")


@app.function(image=image, memory=1024 * 16, timeout=7200, retries=2, cpu=4.0, volumes=VOLUMES)
def block_hash_shard(
    root: str,
    start: int,
    end: int,
    tag: str,
    user_key_column: str | None,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Tokenize + block-hash rows ``[start, end)``; write one pickle shard."""
    import os
    import pickle
    import time

    import tiktoken

    out_dir = os.path.join(TMP_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"shard_{start:09d}.pkl")
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            rows = pickle.load(f)
        return {"start": start, "rows": len(rows), "skipped": True}

    enc = tiktoken.encoding_for_model("gpt-4o")
    dset = _load_split(root)
    columns = set(dset.column_names)
    message_source = _dedup_message_source(columns)

    t0 = time.time()
    records: list[tuple[str, int, array]] = []
    tokens = 0
    batch = SHARD_ROWS  # one slice; row range already small
    block = dset[start:end]
    keys_list = list(block.keys())
    blen = len(block[keys_list[0]]) if keys_list else 0
    for j in range(blen):
        row = {k: block[k][j] for k in keys_list}
        row_index = start + j
        uid = _user_key(row, columns, row_index, user_key_column)
        token_ids = _row_to_token_ids(row, enc, message_source=message_source)
        if not token_ids:
            continue
        records.append((uid, len(token_ids), _block_keys(token_ids, block_size)))
        tokens += len(token_ids)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    hf_datasets_volume.commit()
    return {
        "start": start,
        "rows": len(records),
        "tokens": tokens,
        "elapsed_s": round(time.time() - t0, 1),
        "skipped": False,
    }


@app.function(image=image, memory=1024 * 160, timeout=14400, volumes=VOLUMES)
def reduce_shards(tag: str, block_size: int = BLOCK_SIZE) -> dict:
    """Stream shards in dataset order; count global/intra/cross block reuse."""
    import os
    import pickle
    import time
    from collections import defaultdict

    out_dir = os.path.join(TMP_ROOT, tag)
    shards = sorted(f for f in os.listdir(out_dir) if f.endswith(".pkl"))
    print(f"[reduce] {len(shards)} shards for {tag}")

    global_seen: set[int] = set()
    intra_seen: dict[str, set[int]] = defaultdict(set)
    user_tokens: dict[str, int] = defaultdict(int)
    users: set[str] = set()
    n_req = 0
    total_tok = 0
    total_blocks = 0
    g_saved = 0
    i_saved = 0
    t0 = time.time()

    for si, fname in enumerate(shards, start=1):
        with open(os.path.join(out_dir, fname), "rb") as f:
            records = pickle.load(f)
        for uid, inp, keys in records:
            n_req += 1
            total_tok += inp
            users.add(uid)
            user_tokens[uid] += inp
            n = len(keys)
            total_blocks += n
            us = intra_seen[uid]
            for bi, key in enumerate(keys):
                bt = block_size if bi < n - 1 else inp - block_size * (n - 1)
                if key in global_seen:
                    g_saved += bt
                else:
                    global_seen.add(key)
                if key in us:
                    i_saved += bt
                else:
                    us.add(key)
        if si % 20 == 0 or si == len(shards):
            print(
                f"  {si}/{len(shards)} | {n_req:,} req | "
                f"{len(global_seen):,} unique blocks | {time.time() - t0:.0f}s",
                flush=True,
            )

    top10 = sorted(user_tokens.values(), reverse=True)[:10]
    out = {
        "tag": tag,
        "requests": n_req,
        "users": len(users),
        "total_input_tokens": total_tok,
        "avg_input_tokens": round(total_tok / n_req, 1) if n_req else 0,
        "requests_per_user": round(n_req / len(users), 1) if users else 0,
        "top10_user_concentration_pct": round(100 * sum(top10) / total_tok, 2) if total_tok else 0,
        "total_blocks": total_blocks,
        "unique_blocks": len(global_seen),
        "block_global_reuse_pct": round(100 * g_saved / total_tok, 2) if total_tok else 0,
        "block_intra_user_reuse_pct": round(100 * i_saved / total_tok, 2) if total_tok else 0,
        "block_cross_user_reuse_pct": round(100 * (g_saved - i_saved) / total_tok, 2)
        if total_tok
        else 0,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print("[reduce] RESULT " + json.dumps(out), flush=True)
    return out


@app.function(image=image, memory=1024 * 8, timeout=14400, volumes=VOLUMES)
def driver(
    preset: str,
    dataset_disk_path: str | None,
    user_key_column: str | None,
    max_rows: int | None,
    block_size: int,
) -> dict:
    import time

    if dataset_disk_path:
        root = _normalize_hf_disk_root(dataset_disk_path)
    else:
        _, root = _find_disk_root(_merge_disk_candidates(preset, None))
    dset = _load_split(root)
    n_total = len(dset)
    limit = n_total if max_rows is None else min(n_total, max_rows)
    tag = preset.lower().strip()
    print(f"[driver] {root} | rows={n_total:,} processing={limit:,} | tag={tag}")

    ranges = [
        (root, s, min(s + SHARD_ROWS, limit), tag, user_key_column, block_size)
        for s in range(0, limit, SHARD_ROWS)
    ]
    print(f"[driver] phase 1: {len(ranges)} shards x {SHARD_ROWS} rows", flush=True)

    t0 = time.time()
    done = 0
    for r in block_hash_shard.starmap(ranges):
        done += 1
        if done % 25 == 0 or done == len(ranges):
            print(f"  phase1 {done}/{len(ranges)} shards", flush=True)
    print(f"[driver] phase 1 done in {time.time() - t0:.0f}s; reducing...", flush=True)

    result = reduce_shards.remote(tag, block_size)
    result["dataset_root"] = root
    result["rows_processed"] = limit
    return result


@app.local_entrypoint()
def run(
    preset: str = "wildchat",
    dataset_disk_path: str | None = None,
    user_key_column: str | None = None,
    max_rows: int | None = None,
    block_size: int = BLOCK_SIZE,
    out_path: str = "",
):
    import os

    result = driver.remote(
        preset=preset,
        dataset_disk_path=dataset_disk_path,
        user_key_column=user_key_column,
        max_rows=max_rows,
        block_size=block_size,
    )
    print(json.dumps(result, indent=2))
    if not out_path:
        out_path = f"results/decoded_v9/block_reuse_{preset.lower().strip()}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


@app.local_entrypoint()
def reduce_only(tag: str = "wildchat", block_size: int = BLOCK_SIZE, out_path: str = ""):
    """Re-run only the reducer over already-written phase-1 shards (cached on volume)."""
    import os

    result = reduce_shards.remote(tag, block_size)
    print(json.dumps(result, indent=2))
    if not out_path:
        out_path = f"results/decoded_v9/block_reuse_{tag}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
