"""fo-vllm: Universal VLM inference via vLLM for FiftyOne."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .operators import CheckVLLMStatus, VLLMInference

if TYPE_CHECKING:
    from fiftyone.plugins.context import PluginContext


def register(plugin: PluginContext) -> None:
    """Register plugin operators with the FiftyOne plugin system."""
    plugin.register(VLLMInference)
    plugin.register(CheckVLLMStatus)
