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

    def list_models(self):
        """Query available models from the vLLM server."""
        if not hasattr(self, "_sync_client"):
            from openai import OpenAI

            self._sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return [m.id for m in self._sync_client.models.list().data]

    def validate_connection(self):
        """Test server connectivity. Raises on failure."""
        models = self.list_models()
        if not models:
            raise ConnectionError("vLLM server returned no models")

    def infer_batch(self, messages, structured_outputs):
        """Run batch inference with structured output constraints.

        Args:
            messages: list of OpenAI-format message lists, one per sample.
            structured_outputs: dict passed to vLLM's StructuredOutputsParams.
                Every task provides this — it is never None. Examples:
                  {"choice": ["cat", "dog"]}
                  {"json": {<JSON schema>}}

        Returns list of response strings (valid JSON or choice-constrained
        string).
        """
        return _run_async(self._async_infer_batch(messages, structured_outputs))

    async def _async_infer_batch(self, messages, structured_outputs):
        extra_body = {"structured_outputs": structured_outputs}
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _call(msgs):
            async with sem:
                resp = await self._aclient.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content

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
