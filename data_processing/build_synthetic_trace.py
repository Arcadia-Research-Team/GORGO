"""Build synthetic Mooncake traces from production metadata only.

Reads tokenized parquet metadata (token_hash, per-message token counts,
timestamps) and generates random token IDs that preserve the multi-turn
prefix reuse structure. No real message content is used or stored.

The output format matches ``build_mooncake_trace.py`` exactly — the
proxy replays synthetic traces identically to real ones.

Usage::

    # Extract metadata + generate synthetic trace for W1 window:
    modal run --env=alessio-dev data_processing/build_synthetic_trace.py::main \\
        --start-time 2026-04-02T00:30:00 --end-time 2026-04-02T01:00:00 \\
        --output-path /data/mooncake_traces/synthetic/prod_synthetic_apr2_0030_to_0100.jsonl \\
        --max-input-tokens 24000

    # W2a nighttime:
    modal run --env=alessio-dev data_processing/build_synthetic_trace.py::main \\
        --start-time 2026-04-02T01:00:00 --end-time 2026-04-02T01:30:00 \\
        --output-path /data/mooncake_traces/synthetic/prod_synthetic_apr2_0100_to_0130.jsonl \\
        --max-input-tokens 24000

    # W2b midday:
    modal run --env=alessio-dev data_processing/build_synthetic_trace.py::main \\
        --start-time 2026-04-02T12:30:00 --end-time 2026-04-02T13:00:00 \\
        --output-path /data/mooncake_traces/synthetic/prod_synthetic_apr2_1230_to_1300.jsonl \\
        --max-input-tokens 24000
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone

import modal

from app import app, completions_volume

DEFAULT_BLOCK_SIZE = 256
DEFAULT_VOCAB_SIZE = 151643  # gpt-4o / Qwen tokenizer vocab

image = (
    modal.Image.debian_slim()
    .pip_install("duckdb", "tiktoken", "pyarrow")
    .add_local_python_source("app", "gorgo")
)


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return ""


def _to_naive_dt(ts):
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if isinstance(ts, str):
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(ts[:26], fmt)
            except ValueError:
                continue
    return None


@app.function(
    image=image,
    memory=1024 * 16,
    timeout=3600,
    volumes={"/data": completions_volume},
)
def build_synthetic_trace(
    start_time: str,
    end_time: str,
    output_path: str,
    max_input_tokens: int = 24000,
    block_size: int = DEFAULT_BLOCK_SIZE,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = 42,
    max_output_tokens: int = 128,
):
    import duckdb
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o")
    rng = random.Random(seed)
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)

    FILE_PREFIX = "llm_responses_202604"
    FILE_CUTOFF = "llm_responses_20260408"

    files = sorted(
        f
        for f in os.listdir("/data")
        if f.endswith(".parquet") and FILE_PREFIX in f and f < FILE_CUTOFF
    )
    print(f"[synthetic] {len(files)} parquet files, window {start_time} -> {end_time}")

    # Phase 1: Extract metadata — per-request (token_hash, timestamp, per_message_tokens)
    t0 = time.perf_counter()
    metadata: list[dict] = []

    con = duckdb.connect()
    for filename in files:
        path = os.path.join("/data", filename)
        cursor = con.execute(
            """
            SELECT
                uuid,
                timestamp,
                request_metadata.token_hash AS token_hash,
                request,
                response
            FROM read_parquet(?)
            WHERE request NOT LIKE '%keep-alive%'
            ORDER BY timestamp
            """,
            [path],
        )
        while True:
            chunk = cursor.fetchmany(2048)
            if not chunk:
                break
            for uuid, ts, token_hash, request_raw, response_raw in chunk:
                ts_dt = _to_naive_dt(ts)
                if ts_dt is None:
                    continue
                if ts_dt < start_dt:
                    continue
                if ts_dt >= end_dt:
                    break

                try:
                    req = json.loads(request_raw) if isinstance(request_raw, str) else request_raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(req, dict):
                    continue

                msgs = req.get("messages", [])
                if not isinstance(msgs, list) or not msgs:
                    continue

                # Extract per-message metadata: role + token count (NO content)
                per_msg = []
                total_tokens = 0
                for msg in msgs:
                    if isinstance(msg, str):
                        n = len(enc.encode(msg, disallowed_special=()))
                        per_msg.append({"role": "raw", "tokens": n})
                        total_tokens += n
                    elif isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        text = _content_to_str(msg.get("content"))
                        n = len(enc.encode(text, disallowed_special=())) if text else 0
                        per_msg.append({"role": role, "tokens": n})
                        total_tokens += n

                if total_tokens == 0 or total_tokens > max_input_tokens:
                    continue

                # Get output length from response
                output_length = 0
                try:
                    resp = (
                        json.loads(response_raw) if isinstance(response_raw, str) else response_raw
                    )
                    if isinstance(resp, dict):
                        usage = resp.get("usage")
                        if isinstance(usage, dict):
                            ct = usage.get("completion_tokens")
                            if isinstance(ct, int) and ct >= 0:
                                output_length = min(ct, max_output_tokens)
                except (json.JSONDecodeError, TypeError):
                    pass

                metadata.append(
                    {
                        "uuid": uuid,
                        "ts_dt": ts_dt,
                        "token_hash": token_hash or "",
                        "per_msg": per_msg,
                        "total_tokens": total_tokens,
                        "output_length": output_length,
                    }
                )

            # Check if we've passed the end time
            if metadata and metadata[-1]["ts_dt"] >= end_dt:
                break

        print(
            f"[synthetic]   {filename}: {len(metadata)} rows so far ({time.perf_counter() - t0:.1f}s)"
        )
    con.close()

    metadata.sort(key=lambda r: r["ts_dt"])
    print(
        f"[synthetic] extracted {len(metadata)} requests metadata in {time.perf_counter() - t0:.1f}s"
    )

    # Phase 2: Generate synthetic token IDs preserving prefix structure
    # For each user, maintain a running conversation prefix.
    # New turns reuse the prefix and append fresh random tokens.
    user_prefixes: dict[str, list[int]] = {}
    hash_to_id: dict[bytes, int] = {}

    def _block_ids(token_ids: list[int]) -> list[int]:
        if not token_ids:
            return []
        ids: list[int] = []
        prev_digest = b""
        for i in range(0, len(token_ids), block_size):
            block = token_ids[i : i + block_size]
            h = hashlib.sha256()
            h.update(prev_digest)
            h.update(b"".join(t.to_bytes(4, "little", signed=False) for t in block))
            digest = h.digest()
            mapped = hash_to_id.get(digest)
            if mapped is None:
                mapped = len(hash_to_id)
                hash_to_id[digest] = mapped
            ids.append(mapped)
            prev_digest = digest
        return ids

    def _gen_random_tokens(n: int) -> list[int]:
        return [rng.randint(0, vocab_size - 1) for _ in range(n)]

    rows_out: list[dict] = []
    first_ts: datetime | None = None

    for entry in metadata:
        token_hash = entry["token_hash"]
        per_msg = entry["per_msg"]

        # Get or create user prefix
        prefix = user_prefixes.get(token_hash, [])

        # Build synthetic token sequence preserving multi-turn structure:
        # Reuse prefix tokens from previous turns, generate new for fresh content
        synthetic_ids: list[int] = []
        prefix_offset = 0

        for msg in per_msg:
            n = msg["tokens"]
            if n == 0:
                continue

            if prefix_offset < len(prefix):
                # Reuse existing prefix tokens (cache hit)
                reuse_count = min(n, len(prefix) - prefix_offset)
                synthetic_ids.extend(prefix[prefix_offset : prefix_offset + reuse_count])
                prefix_offset += reuse_count
                remaining = n - reuse_count
                if remaining > 0:
                    fresh = _gen_random_tokens(remaining)
                    synthetic_ids.extend(fresh)
                    prefix_offset += remaining
            else:
                # All new content
                fresh = _gen_random_tokens(n)
                synthetic_ids.extend(fresh)
                prefix_offset += n

        # Update user prefix for next turn
        user_prefixes[token_hash] = synthetic_ids

        # Generate block IDs (prefix-aware hashing)
        block_ids = _block_ids(synthetic_ids)

        # Build synthetic request body for proxy replay.
        # Decode the synthetic token IDs back to text per message so SGLang
        # re-tokenizes to exactly the right length.
        synthetic_messages = []
        tok_offset = 0
        for msg in per_msg:
            n = msg["tokens"]
            msg_token_ids = synthetic_ids[tok_offset : tok_offset + n]
            tok_offset += n
            synthetic_text = enc.decode(msg_token_ids) if msg_token_ids else ""
            synthetic_messages.append(
                {
                    "role": msg["role"],
                    "content": synthetic_text,
                }
            )

        ts_dt = entry["ts_dt"]
        if first_ts is None:
            first_ts = ts_dt
        delta_ms = int((ts_dt - first_ts).total_seconds() * 1000)

        request_id = f"synthetic_{ts_dt.strftime('%Y%m%dT%H%M%S')}_{len(rows_out):06d}"
        output_length = entry["output_length"] or rng.randint(16, 128)

        row = {
            "timestamp": delta_ms,
            "input_length": len(synthetic_ids),
            "output_length": output_length,
            "unique_input_tokens": len(synthetic_ids),  # will be recalculated during replay
            "hash_ids": block_ids,
            "request": {
                "model": "",
                "messages": synthetic_messages,
                "max_tokens": max_output_tokens,
                "stream": True,
            },
            "response": None,
            "request_id": request_id,
            "token_hash": token_hash,
        }
        rows_out.append(row)

    # Phase 3: Write trace JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row, default=str) + "\n")
    os.replace(tmp_path, output_path)
    completions_volume.commit()

    total_input = sum(r["input_length"] for r in rows_out)
    unique_blocks = len(hash_to_id)
    total_blocks = sum(len(r["hash_ids"]) for r in rows_out)
    duration_ms = rows_out[-1]["timestamp"] if rows_out else 0

    print(f"\n[synthetic] wrote {output_path}")
    print(f"  requests: {len(rows_out)}")
    print(f"  users: {len(user_prefixes)}")
    print(f"  total input tokens: {total_input:,}")
    print(
        f"  unique blocks: {unique_blocks:,} / {total_blocks:,} ({100 * unique_blocks / max(total_blocks, 1):.1f}% unique)"
    )
    print(f"  duration: {duration_ms / 1000:.0f}s")
    print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

    return {
        "output_path": output_path,
        "requests": len(rows_out),
        "users": len(user_prefixes),
        "total_input_tokens": total_input,
        "unique_blocks": unique_blocks,
        "total_blocks": total_blocks,
        "duration_ms": duration_ms,
    }


@app.local_entrypoint()
def main(
    start_time: str = "2026-04-02T00:30:00",
    end_time: str = "2026-04-02T01:00:00",
    output_path: str = "",
    max_input_tokens: int = 24000,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = 42,
):
    if not output_path:
        st = start_time.replace(":", "").replace("-", "").replace("T", "_")
        et = end_time.replace(":", "").replace("-", "").replace("T", "_")
        output_path = f"/data/mooncake_traces/synthetic/prod_synthetic_{st}_to_{et}.jsonl"

    result = build_synthetic_trace.remote(
        start_time=start_time,
        end_time=end_time,
        output_path=output_path,
        max_input_tokens=max_input_tokens,
        block_size=block_size,
        seed=seed,
    )
    print(json.dumps(result, indent=2))
