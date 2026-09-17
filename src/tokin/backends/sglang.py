from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, ClassVar

import httpx2

from .backend import Generation, GenerationBackend, GenerationError, GenerationParams


class SGLangBackend(GenerationBackend):
    """Sglang's native `/generate` endpoint."""

    param_map: ClassVar[dict[str, str]] = {
        "max_tokens": "max_new_tokens",
        "stop_ids": "stop_token_ids",
        "seed": "sampling_seed",
        "response_schema": "json_schema",
        "return_logprobs": "return_logprob",
        "request_id": "rid",
    }
    param_convert: ClassVar[dict[str, Callable[[Any], Any]]] = {"stop_ids": sorted, "response_schema": json.dumps}
    # These sit at the top level of the `/generate` body; every other key goes into `sampling_params`.
    top_level_keys: ClassVar[frozenset[str]] = frozenset({"return_logprob", "rid"})

    def __init__(self, url: str, client: httpx2.AsyncClient, *, retries: int = 60, retry_delay: float = 1.0) -> None:
        self.url = url.rstrip("/")
        self.client = client
        self.retries = retries
        self.retry_delay = retry_delay

    async def post(self, payload: dict[str, Any]) -> Any:
        """The JSON `/generate` answers, after however many retries it takes."""
        for attempt in range(1, self.retries + 1):
            try:
                response = await self.client.post(f"{self.url}/generate", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx2.HTTPError, ValueError) as e:
                if attempt == self.retries:
                    raise GenerationError(f"sglang failed {attempt} times, last: {e}") from e
                await asyncio.sleep(self.retry_delay)

    async def generate(self, token_ids: list[int], params: GenerationParams) -> Generation:
        payload: dict[str, Any] = {"input_ids": token_ids, "stream": False, "sampling_params": {}}
        for name, value in self.translate(params).items():
            target = payload if name in self.top_level_keys else payload["sampling_params"]
            target[name] = value
        out = await self.post(payload)
        meta = out["meta_info"]
        if meta["finish_reason"]["type"] not in ("stop", "length", "abort"):
            raise GenerationError(f"sglang finished with {meta['finish_reason']}")
        logprobs = [lp for lp, _, _ in meta["output_token_logprobs"]] if params.get("return_logprobs") else None
        return Generation(token_ids=out["output_ids"], finish_reason=meta["finish_reason"]["type"], logprobs=logprobs)
