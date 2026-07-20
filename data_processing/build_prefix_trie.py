"""Build a path-compressed radix trie over prompt token sequences from the
April week-1 dataset, then report intra-user vs cross-user KV-cache prefix
overlap.

Reads cached per-session token-id streams from
``/data/tokenized_<FILE_PREFIX>/*.tokenized.parquet`` (produced by
``build_eval_dataset.tokenize_dataset``). For each session the flat
prompt-id sequence is streamed into:

- one global trie (all users pooled)
- one trie per ``token_hash`` (intra-user only)

Overlap is measured as tokens saved vs. a naive no-sharing baseline:
    saved = total_tokens_inserted - unique_tokens_in_trie
where ``unique_tokens_in_trie`` is the sum of all compressed edge lengths.
"""

import itertools
from array import array

import modal

from app import app, completions_volume
from build_eval_dataset import (
    FILE_CUTOFF,
    FILE_PREFIX,
    tokenize_dataset,
    tokenized_dir,
    tokenized_path_for,
)
from gorgo.radix_trie import RadixTrie

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb")
    .add_local_python_source("app", "build_eval_dataset", "utils")
)


def _fmt_pct(num: float, den: float) -> str:
    if den <= 0:
        return "  n/a"
    return f"{100.0 * num / den:6.2f}%"


@app.function(
    image=image,
    memory=1024 * 32,
    timeout=7200,
    volumes={"/data": completions_volume},
)
def build_tries(batch_size: int = 20, min_sequence_len: int = 1):
    import os
    import pickle
    import sys
    import time

    import duckdb

    tok_dir = tokenized_dir(FILE_PREFIX)
    if not os.path.isdir(tok_dir):
        raise RuntimeError(
            f"Tokenized cache not found at {tok_dir}. "
            f"Run `modal run build_eval_dataset.py::tokenize_main` first."
        )

    raw_files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet") and FILE_PREFIX in f and f < FILE_CUTOFF
    )
    files = []
    missing = []
    for f in raw_files:
        tok_path = tokenized_path_for(f, file_prefix=FILE_PREFIX)
        if os.path.exists(tok_path):
            files.append(tok_path)
        else:
            missing.append(f)
    if missing:
        print(
            f"WARNING: {len(missing)} raw parquets have no tokenized cache "
            f"(will be skipped): {missing[:3]}{'...' if len(missing) > 3 else ''}"
        )
    batches = list(itertools.batched(files, batch_size))
    print(f"April 1-7: {len(files)} tokenized files in {len(batches)} batch(es) of <= {batch_size}")

    # Radix tries can get deep; pickle's default 1000-frame recursion limit is
    # easy to blow through.
    sys.setrecursionlimit(1_000_000)

    tries_dir = f"/data/prefix_tries_{FILE_PREFIX}"
    os.makedirs(tries_dir, exist_ok=True)

    global_trie = RadixTrie()
    per_user_tries: dict[str, RadixTrie] = {}
    total_tokens = 0
    total_sequences = 0
    skipped_empty = 0
    resume_from = -1

    existing_ckpts = sorted(
        f for f in os.listdir(tries_dir) if f.startswith("batch_") and f.endswith(".pkl")
    )
    if existing_ckpts:
        latest = os.path.join(tries_dir, existing_ckpts[-1])
        print(f"Resuming from checkpoint: {latest}")
        with open(latest, "rb") as f:
            payload = pickle.load(f)
        if payload.get("num_batches") != len(batches):
            print(
                f"  WARNING: checkpoint saw {payload.get('num_batches')} batches, "
                f"current run has {len(batches)}; batch indices may not align. "
                f"Delete {tries_dir} to start fresh."
            )
        global_trie = payload["global_trie"]
        per_user_tries = payload["per_user_tries"]
        total_tokens = payload["total_tokens"]
        total_sequences = payload["total_sequences"]
        skipped_empty = payload["skipped_empty"]
        resume_from = payload["batch_idx"]
        print(
            f"  restored: batch {resume_from + 1}/{len(batches)}, "
            f"{total_sequences:,} seqs, {total_tokens:,} toks, "
            f"{len(per_user_tries):,} users"
        )

    t0 = time.time()
    con = duckdb.connect()

    for batch_idx, batch in enumerate(batches):
        if batch_idx <= resume_from:
            continue
        batch_sequences = 0
        batch_tokens = 0
        for tok_path in batch:
            cursor = con.execute(
                "SELECT token_hash, prompt_ids FROM read_parquet(?)",
                [tok_path],
            )
            while True:
                chunk = cursor.fetchmany(2048)
                if not chunk:
                    break
                for token_hash, token_ids in chunk:
                    if not token_ids or len(token_ids) < min_sequence_len:
                        skipped_empty += 1
                        continue
                    seq = array("I", token_ids)

                    global_trie.insert(seq)
                    user_trie = per_user_tries.get(token_hash)
                    if user_trie is None:
                        user_trie = RadixTrie()
                        per_user_tries[token_hash] = user_trie
                    user_trie.insert(seq)

                    total_tokens += len(seq)
                    total_sequences += 1
                    batch_sequences += 1
                    batch_tokens += len(seq)

        elapsed = time.time() - t0
        print(
            f"  batch {batch_idx + 1}/{len(batches)}: "
            f"+{batch_sequences:,} seqs (+{batch_tokens:,} toks) | "
            f"cumulative {total_sequences:,} seqs, {total_tokens:,} toks, "
            f"{len(per_user_tries):,} users | elapsed {elapsed:,.0f}s"
        )

        ckpt_path = os.path.join(tries_dir, f"batch_{batch_idx:04d}.pkl")
        tmp_path = ckpt_path + ".tmp"
        payload = {
            "batch_idx": batch_idx,
            "num_batches": len(batches),
            "files_processed": list(itertools.chain.from_iterable(batches[: batch_idx + 1])),
            "total_sequences": total_sequences,
            "total_tokens": total_tokens,
            "skipped_empty": skipped_empty,
            "global_trie": global_trie,
            "per_user_tries": per_user_tries,
        }
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, ckpt_path)
        completions_volume.commit()
        ckpt_bytes = os.path.getsize(ckpt_path)
        print(f"    checkpoint: {ckpt_path} ({ckpt_bytes / 1e6:,.1f} MB)")

    if total_sequences == 0:
        print("No sequences ingested; nothing to report.")
        return

    print("\nComputing overlap stats...")
    global_unique = global_trie.unique_token_count()

    sum_intra_unique = 0
    user_count = 0
    top_users = []  # (sum_tokens, unique, user)
    for token_hash, user_trie in per_user_tries.items():
        u = user_trie.unique_token_count()
        sum_intra_unique += u
        user_count += 1
        top_users.append((user_trie.total_tokens_inserted, u, token_hash))

    intra_savings = total_tokens - sum_intra_unique
    global_savings = total_tokens - global_unique
    cross_user_extra = sum_intra_unique - global_unique

    print(f"\n{'=' * 72}")
    print(f"Sequences inserted:            {total_sequences:>14,}")
    print(f"Unique users (token_hash):     {user_count:>14,}")
    print(f"Empty sequences skipped:       {skipped_empty:>14,}")
    print(f"Total tokens T:                {total_tokens:>14,}")
    print(f"{'-' * 72}")
    print("(A) INTRA-USER overlap   -- prefixes shared within a user's own requests")
    print(
        f"      (T - sum_u U(R_u)) / T   "
        f"= {intra_savings:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(intra_savings, total_tokens)}"
    )
    print()
    print("(C) CROSS-USER overlap   -- extra sharing from pooling across users")
    print(
        f"      (sum_u U(R_u) - U(all)) / T   "
        f"= {cross_user_extra:>14,} / {total_tokens:,}  "
        f"= {_fmt_pct(cross_user_extra, total_tokens)}"
    )
    print(f"{'-' * 72}")
    print(f"Sanity check: A + C = global overlap (B)")
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

    top_users.sort(reverse=True)
    print("\nTop 10 users by tokens inserted:")
    print(f"  {'tokens':>14}  {'unique':>14}  {'savings':>8}  token_hash")
    for total_t, uniq_t, th in top_users[:10]:
        saved = total_t - uniq_t
        print(f"  {total_t:>14,}  {uniq_t:>14,}  {_fmt_pct(saved, total_t)}  {th}")

    import json

    stats = {
        "batch_size": batch_size,
        "min_sequence_len": min_sequence_len,
        "file_prefix": FILE_PREFIX,
        "file_cutoff": FILE_CUTOFF,
        "num_files": len(files),
        "num_batches": len(batches),
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "skipped_empty": skipped_empty,
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
        "top_users": [
            {"token_hash": th, "tokens": total_t, "unique_tokens": uniq_t}
            for total_t, uniq_t, th in top_users[:10]
        ],
    }

    stats_path = f"/data/prefix_trie_stats_{FILE_PREFIX}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    completions_volume.commit()
    print(f"\nStats written to {stats_path}")


@app.local_entrypoint()
def trie_main(batch_size: int = 20, min_sequence_len: int = 1):
    print("Step 1/2: ensuring tokenized cache is complete...")
    tok_summary = tokenize_dataset.remote(file_prefix=FILE_PREFIX, file_cutoff=FILE_CUTOFF)
    print(
        f"Tokenization done: {tok_summary['num_sessions']:,} sessions across "
        f"{tok_summary['num_files']:,} files "
        f"({tok_summary['skipped_files']:,} already cached)."
    )

    print("\nStep 2/2: building tries...")
    build_tries.remote(batch_size=batch_size, min_sequence_len=min_sequence_len)
