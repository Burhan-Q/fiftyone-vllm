"""fo-vllm: Universal VLM inference via vLLM for FiftyOne."""

from .operators import VLLMInference


def register(plugin):
    plugin.register(VLLMInference)
