"""Image loading/encoding utilities for vLLM chat messages."""

import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor


def build_image_contents(
    filepaths,
    image_mode="auto",
    base_url=None,
    max_workers=8,
):
    """Build image content dicts for vLLM chat messages.

    Returns OpenAI-format image_url content dicts. All I/O-bound
    work is parallelized via ThreadPoolExecutor.

    Args:
        filepaths: absolute paths to image files.
        image_mode: "auto" | "base64" | "filepath"
            - "auto": filepath for local servers; base64 for remote
            - "filepath": file:// URL strings (local servers only, no I/O)
            - "base64": base64 data URIs (works everywhere, parallel I/O)
        base_url: vLLM server URL, used by "auto" to detect local servers.
        max_workers: ThreadPoolExecutor size for parallel encoding.
    """
    resolved = _resolve_image_mode(image_mode, base_url)

    if resolved == "filepath":
        return [_filepath_content(fp) for fp in filepaths]
    elif resolved == "base64":
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_encode_base64, filepaths))
    else:
        raise ValueError(f"Unknown image mode: {resolved}")


def _resolve_image_mode(image_mode, base_url):
    """Resolve the effective image mode from user settings and context."""
    if image_mode != "auto":
        return image_mode
    if base_url and any(
        h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")
    ):
        return "filepath"
    return "base64"


def _filepath_content(filepath):
    """Build file:// URL content dict (no I/O)."""
    abspath = os.path.abspath(filepath)
    return {
        "type": "image_url",
        "image_url": {"url": f"file://{abspath}"},
    }


def _encode_base64(filepath):
    """Read and base64-encode a single image file."""
    mime = mimetypes.guess_type(filepath)[0] or "image/jpeg"
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }
