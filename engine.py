"""VLLMEngine: online inference via AsyncOpenAI with structured output."""

import asyncio
import concurrent.futures


class VLLMEngine:
    """Thin wrapper over vLLM's OpenAI-compatible API with structured output."""

    def __init__(
        self,
        model,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        max_concurrent=16,
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        seed=None,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.max_concurrent = max_concurrent
        self._aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._sync_client = None

    def list_models(self):
        """Query available models from the vLLM server."""
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return [m.id for m in self._sync_client.models.list().data]

    def validate_connection(self):
        """Test server connectivity. Raises on failure."""
        models = self.list_models()
        if not models:
            raise ConnectionError("vLLM server returned no models")

    def infer_batch(self, messages, response_model):
        """Run batch inference with Pydantic structured output.

        Args:
            messages: list of OpenAI-format message lists, one per sample.
            response_model: Pydantic BaseModel class for response parsing.

        Returns list of validated Pydantic instances (or Exceptions).
        """
        return _run_async(self._async_infer_batch(messages, response_model))

    async def _async_infer_batch(self, messages, response_model):
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _call(msgs):
            async with sem:
                resp = await self._aclient.beta.chat.completions.parse(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    response_format=response_model,
                )
                msg = resp.choices[0].message
                if msg.refusal:
                    raise ValueError(f"Model refused: {msg.refusal}")
                if msg.parsed is None:
                    raise ValueError(f"Parsing failed: {msg.content}")
                return msg.parsed

        results = await asyncio.gather(
            *[_call(m) for m in messages], return_exceptions=True
        )
        return list(results)


def _run_async(coro):
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
