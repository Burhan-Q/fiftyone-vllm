"""Image loading/encoding utilities for vLLM chat messages."""

import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor


def build_image_contents(filepaths, image_mode="auto", max_workers=4):
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
    results = [None] * len(filepaths)
    to_encode = []

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


def _url_content(url):
    """Wrap an HTTP(S) URL as an image_url content dict."""
    return {"type": "image_url", "image_url": {"url": url}}


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
