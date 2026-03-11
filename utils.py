"""Image loading/encoding utilities and config persistence for vLLM."""

from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
from concurrent.futures import Executor
from functools import partial
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from fiftyone.operators.store import ExecutionStore

ContentPart = dict[str, str | dict[str, str]]
"""Single content part: text block or image_url block."""

_MIN_IMAGE_DIM = 512
"""Minimum image dimension — below this VLMs produce garbage."""

_MIN_IMAGE_BUDGET_TOKENS = 400
"""Minimum image token budget before falling back to _MIN_IMAGE_DIM."""

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
    "max_image_dim",
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


def default_max_image_dim(
    max_model_len: int | None,
    prompt_text: str | None = None,
    system_text: str | None = None,
    max_tokens: int = 512,
) -> int | None:
    """Derive max image dimension from model context length.

    Estimates prompt token usage, reserves space for output generation,
    and allocates the remaining budget to image tokens. Uses Qwen2.5-VL's
    28x28 patch tokenization (784 pixels/token) as baseline.

    Returns None if max_model_len is unknown (no resizing).
    """
    if max_model_len is None:
        return None

    # Estimate text tokens: chars/4 heuristic + 64 token framing margin
    text_tokens = sum(len(t) // 4 for t in (prompt_text or "", system_text or "") if t)
    overhead = text_tokens + max_tokens + 64

    image_budget = max_model_len - overhead
    if image_budget < _MIN_IMAGE_BUDGET_TOKENS:
        return _MIN_IMAGE_DIM

    dim = int((image_budget * 28 * 28) ** 0.5)
    return max(_MIN_IMAGE_DIM, dim)  # No upper cap — model context is the natural limit


def _resize_and_encode_base64(filepath: str, max_dim: int) -> ContentPart:
    """Read, resize if needed, and base64-encode a single image file."""

    img = Image.open(filepath)

    # Fast path: skip PIL decode/re-encode if image already fits and needs
    # no EXIF rotation.  getexif() and .size read from the header only.
    needs_resize = max(img.width, img.height) > max_dim
    orientation = img.getexif().get(0x0112, 1)  # 274 = Orientation tag
    needs_rotate = orientation not in (1, None)

    if not needs_resize and not needs_rotate and img.format == "JPEG":
        img.close()
        return _encode_base64(filepath)

    # Slow path: full decode, optional rotate/resize, re-encode
    img = ImageOps.exif_transpose(img)
    if needs_resize:
        img.thumbnail((max_dim, max_dim), Image.BILINEAR)
    buf = BytesIO()
    if img.mode == "RGBA":
        fmt, mime = "PNG", "image/png"
    else:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        fmt, mime = "JPEG", "image/jpeg"
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _encode_batch(
    paths: list[str],
    max_image_dim: int | None,
    executor: Executor,
) -> list[ContentPart]:
    """Encode a list of local image paths in parallel, optionally resizing."""
    encoder = partial(_resize_and_encode_base64, max_dim=max_image_dim) if max_image_dim is not None else _encode_base64
    return list(executor.map(encoder, paths))


def build_image_contents(
    filepaths: list[str],
    image_mode: str = "auto",
    max_image_dim: int | None = None,
    executor: Executor | None = None,
) -> list[ContentPart]:
    """Build image content dicts for vLLM chat messages.

    URLs (http/https) are always passed through directly.
    Local files are base64-encoded by default, or sent as file:// URLs
    in "filepath" mode (requires --allowed-local-media-path on the
    vLLM server). When max_image_dim is set, local files are always
    resized and base64-encoded (file:// refs cannot be resized).

    Args:
        filepaths: paths or URLs to image files.
        image_mode: "auto" | "filepath"
            - "auto": URLs pass through, local files base64-encoded
            - "filepath": URLs pass through, local files as file:// URLs
        max_image_dim: max pixel size for longest side (None to skip resizing).
        executor: process/thread pool for parallel encoding (caller-owned).
    """
    results: list[ContentPart | None] = [None] * len(filepaths)
    to_encode: list[int] = []

    for i, fp in enumerate(filepaths):
        if fp.startswith(("http://", "https://")):
            results[i] = _url_content(fp)
        elif image_mode == "filepath" and max_image_dim is None:
            results[i] = _filepath_content(fp)
        else:
            to_encode.append(i)

    if to_encode:
        paths = [filepaths[i] for i in to_encode]
        for idx, enc in zip(to_encode, _encode_batch(paths, max_image_dim, executor)):
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


def effective_dims(
    widths: list[int | None],
    heights: list[int | None],
    max_dim: int,
) -> tuple[list[int | None], list[int | None]]:
    """Compute effective image dimensions after thumbnail resize.

    Mirrors PIL's thumbnail aspect-ratio-preserving logic.
    Returns original dims when no resize would occur.
    """
    eff_w: list[int | None] = []
    eff_h: list[int | None] = []
    for w, h in zip(widths, heights):
        if w is not None and h is not None and max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            eff_w.append(int(w * scale))
            eff_h.append(int(h * scale))
        else:
            eff_w.append(w)
            eff_h.append(h)
    return eff_w, eff_h
