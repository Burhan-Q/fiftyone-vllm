"""VLLMEngine: online inference via AsyncOpenAI with structured output."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TypeVar

from openai import AsyncOpenAI, OpenAI

ContentPart = dict[str, str | dict[str, str]]
"""Single content part: text block or image_url block."""

ChatMessage = dict[str, str | list[ContentPart]]
"""One message: {"role": ..., "content": str | [ContentPart, ...]}."""

_T = TypeVar("_T")


@dataclass
class EngineConfig:
    """Configuration for VLLMEngine."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_concurrent: int = 16
    temperature: float | None = None
    max_tokens: int = 512
    top_p: float = 1.0
    seed: int | None = None


class VLLMEngine:
    """Thin wrapper over vLLM's OpenAI-compatible API with structured output."""

    def __init__(self, config: EngineConfig) -> None:
        """Initialize the engine from a config dataclass.

        Args:
            config: engine configuration with model, server, and generation params.
        """
        self.model = config.model
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.seed = config.seed
        self.max_concurrent = config.max_concurrent
        self._aclient = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self._sync_client: OpenAI | None = None

    def list_models(self) -> list[str]:
        """Query available models from the vLLM server."""
        if self._sync_client is None:
            self._sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return [m.id for m in self._sync_client.models.list().data]

    def validate_connection(self) -> None:
        """Test server connectivity. Raises on failure."""
        models = self.list_models()
        if not models:
            raise ConnectionError("vLLM server returned no models")

    def infer_batch(
        self,
        messages: list[list[ChatMessage]],
        structured_outputs: dict[str, object],
    ) -> list[str | Exception]:
        """Run batch inference with structured output constraints.

        Args:
            messages: list of OpenAI-format message lists, one per sample.
            structured_outputs: dict passed to vLLM's StructuredOutputsParams.
                Every task provides this — it is never None. Examples:
                  {"choice": ["cat", "dog"]}
                  {"json": {<JSON schema>}}

        Returns:
            List of response strings (valid JSON or choice-constrained string).
        """
        return _run_async(self._async_infer_batch(messages, structured_outputs))

    async def _async_infer_batch(
        self,
        messages: list[list[ChatMessage]],
        structured_outputs: dict[str, object],
    ) -> list[str | Exception]:
        """Execute batch inference asynchronously with concurrency limiting."""
        extra_body = {"structured_outputs": structured_outputs}
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _call(msgs: list[ChatMessage]) -> str:
            """Send a single chat completion request under the semaphore."""
            async with sem:
                kwargs: dict[str, object] = {
                    "model": self.model,
                    "messages": msgs,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "extra_body": extra_body,
                }
                if self.seed is not None:
                    kwargs["seed"] = self.seed
                resp = await self._aclient.chat.completions.create(**kwargs)
                return resp.choices[0].message.content

        results = await asyncio.gather(*[_call(m) for m in messages], return_exceptions=True)
        return list(results)


def _run_async(coro: Coroutine[object, object, _T]) -> _T:
    """Run an async coroutine safely, handling existing event loops.

    FiftyOne's App runs a Uvicorn server with its own event loop.
    Calling asyncio.run() from within that context raises
    RuntimeError: 'This event loop is already running'.
    This helper detects that case and runs in a dedicated thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
