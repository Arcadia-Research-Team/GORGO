import asyncio
import time

from gorgo.measure import consume_sse_stream


class _FakeSSEResponse:
    def __init__(self, events: list[bytes]):
        self._events = events

    async def aiter_raw(self):
        for e in self._events:
            yield e


def _drain(events: list[bytes]):
    return asyncio.run(
        consume_sse_stream(
            _FakeSSEResponse(events),
            request_start_ns=time.perf_counter_ns(),
        )
    )


def test_usage_with_prompt_tokens_details_cached_tokens():
    events = [
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
        b'data: {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 1,'
        b' "prompt_tokens_details": {"cached_tokens": 64}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    ttft_ns, output_tokens, prompt_tokens, completion_tokens, meta_info, cached = _drain(events)
    assert ttft_ns is not None
    assert output_tokens == 1
    assert prompt_tokens == 100
    assert completion_tokens == 1
    assert cached == 64


def test_usage_without_details_leaves_cached_none():
    events = [
        b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
        b'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}\n\n',
        b"data: [DONE]\n\n",
    ]
    *_, cached = _drain(events)
    assert cached is None


def test_details_with_null_cached_tokens_leaves_cached_none():
    # SGLang emits "cached_tokens_details": null on some paths; guard the
    # analogous null inside prompt_tokens_details.
    events = [
        b'data: {"choices": [], "usage": {"prompt_tokens": 10,'
        b' "prompt_tokens_details": {"cached_tokens": null}}}\n\n',
    ]
    *_, cached = _drain(events)
    assert cached is None
