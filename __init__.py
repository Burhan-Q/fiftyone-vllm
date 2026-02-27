"""fo-vllm: Universal VLM inference via vLLM for FiftyOne."""

import fiftyone.operators as foo

from .operators import VLLMInference


def register(plugin):
    plugin.register(VLLMInference)
