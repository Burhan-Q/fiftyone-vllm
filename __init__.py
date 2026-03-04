"""fo-vllm: Universal VLM inference via vLLM for FiftyOne."""

from .operators import CheckVLLMStatus, VLLMInference


def register(plugin):
    plugin.register(VLLMInference)
    plugin.register(CheckVLLMStatus)
