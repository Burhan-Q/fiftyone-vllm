"""Image loading/encoding utilities and config persistence for vLLM."""

from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fiftyone.operators.store import ExecutionStore

ContentPart = dict[str, str | dict[str, str]]
"""Single content part: text block or image_url block."""

_PERSIST_KEYS = [
    "model",
    "base_url",
    "api_key",
    "task",
    "classes",
    "question",
    "prompt",
    "system_prompt",
    "prompt_override",
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "batch_size",
    "max_concurrent",
    "max_workers",
    "image_mode",
    "coordinate_format",
    "box_format",
]


def normalize_classes(raw: str | list[str] | None) -> list[str] | None:
    """Convert classes from comma-separated string or list to a clean list.

    Returns a list of strings, or None if empty.
    """
    if not raw:
        return None
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    return [c.strip() for c in raw.split(",") if c.strip()] or None


def pick_params(params: dict[str, object], exclude: tuple[str, ...] = ()) -> dict[str, object]:
    """Filter params to persistable keys, dropping Nones.

    Normalizes classes to a list for consistent storage.
    """
    out: dict[str, object] = {}
    for k in _PERSIST_KEYS:
        if k in params and k not in exclude and params[k] is not None:
            v = normalize_classes(params[k]) if k == "classes" else params[k]
            if v is not None:
                out[k] = v
    return out


def _global_store() -> ExecutionStore:
    """Return the cross-dataset ExecutionStore for vLLM config."""
    return ExecutionStore.create("vllm_config", dataset_id=None)


def get_global_config() -> dict[str, object]:
    """Read global config from the cross-dataset ExecutionStore."""
    try:
        cfg = _global_store().get("config")
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def save_global_config(params: dict[str, object]) -> None:
    """Persist params to the cross-dataset ExecutionStore."""
    with contextlib.suppress(Exception):
        _global_store().set("config", pick_params(params))


def clear_global_config() -> None:
    """Delete the cross-dataset config key."""
    with contextlib.suppress(Exception):
        _global_store().delete("config")


def parse_config_json(json_str: str) -> tuple[dict[str, object] | None, str | None]:
    """Parse JSON string, filter to known keys. Returns (dict, None) or (None, err)."""
    try:
        raw = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(raw, dict):
        return None, "JSON must be an object"
    return {k: raw[k] for k in _PERSIST_KEYS if k in raw}, None


def build_image_contents(
    filepaths: list[str],
    image_mode: str = "auto",
    max_workers: int = 4,
) -> list[ContentPart]:
    """Build image content dicts for vLLM chat messages.

    URLs (http/https) are always passed through directly.
    Local files are base64-encoded by default, or sent as file:// URLs
    in "filepath" mode (requires --allowed-local-media-path on the
    vLLM server).

    Args:
        filepaths: paths or URLs to image files.
        image_mode: "auto" | "filepath"
            - "auto": URLs pass through, local files base64-encoded
            - "filepath": URLs pass through, local files as file:// URLs
        max_workers: ThreadPoolExecutor size for parallel base64 encoding.
    """
    results: list[ContentPart | None] = [None] * len(filepaths)
    to_encode: list[int] = []

    for i, fp in enumerate(filepaths):
        if fp.startswith(("http://", "https://")):
            results[i] = _url_content(fp)
        elif image_mode == "filepath":
            results[i] = _filepath_content(fp)
        else:
            to_encode.append(i)

    if to_encode:
        paths = [filepaths[i] for i in to_encode]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            encoded = list(pool.map(_encode_base64, paths))
        for idx, enc in zip(to_encode, encoded):
            results[idx] = enc

    return results


def _url_content(url: str) -> ContentPart:
    """Wrap an HTTP(S) URL as an image_url content dict."""
    return {"type": "image_url", "image_url": {"url": url}}


def _filepath_content(filepath: str) -> ContentPart:
    """Build file:// URL content dict (no I/O)."""
    abspath = str(Path(filepath).resolve())
    return {
        "type": "image_url",
        "image_url": {"url": f"file://{abspath}"},
    }


def _encode_base64(filepath: str) -> ContentPart:
    """Read and base64-encode a single image file."""
    mime = mimetypes.guess_type(filepath)[0] or "image/jpeg"
    with Path(filepath).open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }
