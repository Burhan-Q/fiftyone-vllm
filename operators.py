"""FiftyOne operator for vLLM inference: UI, batching, progress, and result
storage."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.operators.store.notification_service import (
    default_notification_service,
)

from .engine import EngineConfig, VLLMEngine
from .tasks import DetectionConfig, TaskConfig
from .utils import (
    build_image_contents,
    clear_global_config,
    get_global_config,
    normalize_classes,
    parse_config_json,
    pick_params,
    save_global_config,
)

if TYPE_CHECKING:
    from fiftyone.operators.executor import ExecutionContext

    import fiftyone as fo

_DEFAULTS: dict[str, object] = {
    "base_url": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "batch_size": 8,
    "max_tokens": 512,
    "top_p": 1.0,
    "max_concurrent": 16,
    "max_workers": 4,
    "image_mode": "auto",
    "coordinate_format": "pixel",
    "box_format": "xyxy",
}


@dataclass
class _InferenceContext:
    """Bundled state for a single inference run, passed between execute phases."""

    engine: VLLMEngine
    task: TaskConfig
    base_url: str
    api_key: str
    field_name: str
    ids: list[str]
    filepaths: list[str]
    widths: list[int | None]
    heights: list[int | None]
    structured_outputs: dict[str, object]
    batch_size: int
    image_mode: str
    max_workers: int
    log_metadata: bool
    metadata: dict[str, object] | None


class VLLMInference(foo.Operator):
    """Run vLLM inference on a FiftyOne dataset."""

    @property
    def config(self) -> foo.OperatorConfig:
        """Return operator configuration."""
        return foo.OperatorConfig(
            name="run_vllm_inference",
            label="Run vLLM Inference",
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx: ExecutionContext) -> types.Property:
        """Build the operator input form."""
        inputs = types.Object()

        if not ctx.dataset:
            inputs.view("error", types.Error(label="No dataset loaded"))
            return types.Property(inputs)

        stored = _resolve_config(ctx)

        mode_radio = types.RadioGroup(orientation="horizontal")
        mode_radio.add_choice("manual", label="Configure manually")
        mode_radio.add_choice("json", label="Paste JSON config")
        mode_radio.add_choice("reset", label="Reset to defaults")
        inputs.enum(
            "config_mode",
            mode_radio.values(),
            default="manual",
            label="Configuration",
            view=mode_radio,
        )
        config_mode = ctx.params.get("config_mode", "manual")

        if config_mode == "json":
            _json_config_inputs(ctx, inputs)
        elif config_mode == "reset":
            inputs.view(
                "reset_notice",
                types.Notice(label="All stored settings (global and dataset) will be cleared and defaults restored."),
            )
        else:
            _model_selector(inputs, stored)
            _server_settings(inputs, stored)
            task = _task_selector(ctx, inputs, stored)
            _task_settings(inputs, task, stored)
            _output_settings(ctx, inputs, task)
            _advanced_settings(ctx, inputs, stored)

        if config_mode != "reset":
            inputs.view_target(ctx)

        return types.Property(inputs, view=types.View(label="vLLM Inference"))

    def resolve_delegation(self, ctx: ExecutionContext) -> bool | None:
        """Resolve whether execution should be delegated."""
        return ctx.params.get("delegate", None)

    def execute(self, ctx: ExecutionContext) -> Generator[object, None, None]:
        """Run vLLM inference on the target dataset view."""
        params = ctx.params
        mode = params.get("config_mode", "manual")

        if mode == "reset":
            yield from _handle_reset(ctx)
            return

        if mode == "json":
            err = _handle_json_import(params)
            if err:
                yield _error(ctx, err)
                return

        inf, err = _prepare_inference(ctx, params)
        if err:
            yield _error(ctx, err)
            return

        yield from _process_batches(ctx, inf)
        yield from _finalize_run(ctx, inf, params)

    def resolve_output(self, ctx: ExecutionContext) -> types.Property:
        """Build the operator output display."""
        outputs = types.Object()
        outputs.str("summary", label="Summary")

        if ctx.params.get("config_mode") != "reset":
            cfg_json = json.dumps(pick_params(ctx.params, exclude=("api_key",)), indent=2)
            outputs.str(
                "config_export",
                label="Exportable Config (copy to reuse)",
                default=cfg_json,
                view=types.CodeView(language="json", read_only=True),
            )

        return types.Property(outputs, view=types.View(label="Complete"))


class CheckVLLMStatus(foo.Operator):
    """Subscribes to the ``vllm_status`` execution store via MongoDB
    change-stream notifications.  When the delegated worker writes a
    completion signal, this operator fires a toast and reloads the dataset.

    Starts automatically on dataset open and waits up to 10 minutes for
    a signal before silently exiting.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        """Return operator configuration."""
        return foo.OperatorConfig(
            name="check_vllm_status",
            label="Check vLLM Status",
            on_dataset_open=True,
            execute_as_generator=True,
            unlisted=True,
        )

    async def execute(self, ctx: ExecutionContext) -> Generator[object, None, None]:
        """Listen for delegated inference completion and notify the user."""
        if not ctx.dataset:
            return

        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        def _on_change(_message: object) -> None:
            """Signal the event when a store change is detected."""
            loop.call_soon_threadsafe(event.set)

        sub_id = default_notification_service.subscribe(
            "vllm_status",
            callback=_on_change,
            dataset_id=str(ctx.dataset._doc.id),
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=600)

            store = ctx.store("vllm_status")
            store.delete("done")

            yield ctx.trigger(
                "notify",
                params={
                    "message": "vLLM inference complete",
                    "variant": "success",
                },
            )
            yield ctx.trigger("reload_dataset")
        except asyncio.TimeoutError:
            pass
        finally:
            default_notification_service.unsubscribe(sub_id)


# -- Execute phase helpers --


def _handle_reset(ctx: ExecutionContext) -> Generator[object, None, None]:
    """Reset mode: clear all stored configs and reload the dataset."""
    clear_global_config()
    ctx.dataset.info.pop("_vllm_config", None)
    ctx.dataset.save()
    if not ctx.delegated:
        yield ctx.trigger("reload_dataset")


def _handle_json_import(params: dict[str, object]) -> str | None:
    """Parse JSON config and merge into params in place.

    Returns an error string on failure, None on success.
    """
    raw = params.get("config_json") or ""
    cfg, err = parse_config_json(raw)
    if err:
        return f"Config import failed: {err}"
    params.update(cfg)
    if not params.get("model") or not params.get("task"):
        return "Config missing required 'model' or 'task'"
    return None


def _prepare_inference(
    ctx: ExecutionContext,
    params: dict[str, object],
) -> tuple[_InferenceContext | None, str | None]:
    """Build engine, task, and collect sample data for inference.

    Returns (context, None) on success or (None, error_message) on failure.
    """
    try:
        engine, base_url, api_key = _create_engine(params, ctx.secrets)
        task = _create_task(params)
    except Exception as e:
        return None, str(e)

    engine.temperature = params.get("temperature") or task.default_temperature or 0.1

    batch_size = params.get("batch_size", _DEFAULTS["batch_size"])
    image_mode = params.get("image_mode", _DEFAULTS["image_mode"])
    max_workers = params.get("max_workers", _DEFAULTS["max_workers"])

    view = ctx.target_view()
    ids = view.values("id")
    filepaths = view.values("filepath")
    total = len(ids)

    structured_outputs = task.get_structured_outputs()
    field_name = _resolve_field_name(ctx.dataset, params["task"], params.get("overwrite_last", False))

    log_metadata = params.get("log_metadata", False)
    metadata = _build_metadata(params, engine, task, batch_size, image_mode) if log_metadata else None

    _clear_stale_errors(ctx.dataset, field_name, ids, params.get("overwrite_last", False))

    need_dims = task.task == "detect" and task.coordinate_format == "pixel"
    if need_dims:
        view.compute_metadata()
        widths = view.values("metadata.width")
        heights = view.values("metadata.height")
    else:
        widths = [None] * total
        heights = [None] * total

    inf = _InferenceContext(
        engine=engine,
        task=task,
        base_url=base_url,
        api_key=api_key,
        field_name=field_name,
        ids=ids,
        filepaths=filepaths,
        widths=widths,
        heights=heights,
        structured_outputs=structured_outputs,
        batch_size=batch_size,
        image_mode=image_mode,
        max_workers=max_workers,
        log_metadata=log_metadata,
        metadata=metadata,
    )
    return inf, None


def _process_batches(
    ctx: ExecutionContext,
    inf: _InferenceContext,
) -> Generator[object, None, None]:
    """Run batch inference loop with progress reporting."""
    total = len(inf.ids)
    processed = 0
    total_errors = 0

    for i in range(0, total, inf.batch_size):
        batch_ids = inf.ids[i : i + inf.batch_size]
        batch_paths = inf.filepaths[i : i + inf.batch_size]
        batch_widths = inf.widths[i : i + inf.batch_size]
        batch_heights = inf.heights[i : i + inf.batch_size]

        image_contents = build_image_contents(
            batch_paths,
            image_mode=inf.image_mode,
            max_workers=inf.max_workers,
        )
        batch_messages = [inf.task.build_messages(img) for img in image_contents]
        responses = inf.engine.infer_batch(batch_messages, structured_outputs=inf.structured_outputs)

        results, errors, batch_errors = _parse_batch_responses(
            inf,
            batch_ids,
            responses,
            batch_widths,
            batch_heights,
        )
        total_errors += batch_errors

        _write_batch_results(ctx.dataset, inf.field_name, results, errors)

        processed += len(batch_ids)
        progress_label = f"{processed}/{total} samples"
        if total_errors:
            progress_label += f" ({total_errors} errors)"

        if ctx.delegated:
            ctx.set_progress(progress=processed / total, label=progress_label)
        else:
            yield ctx.trigger("set_progress", {"progress": processed / total, "label": progress_label})


def _finalize_run(
    ctx: ExecutionContext,
    inf: _InferenceContext,
    params: dict[str, object],
) -> Generator[object, None, None]:
    """Persist run metadata, save config, and notify/reload."""
    if inf.log_metadata and inf.metadata:
        runs = ctx.dataset.info.get("vllm_runs", {})
        runs[inf.field_name] = inf.metadata
        ctx.dataset.info["vllm_runs"] = runs

    params["base_url"] = inf.base_url
    params["api_key"] = inf.api_key
    save_global_config(params)
    ctx.dataset.info["_vllm_config"] = pick_params(params)
    ctx.dataset.save()

    if ctx.delegated:
        store = ctx.store("vllm_status")
        store.set("done", True)
    else:
        yield ctx.trigger("reload_dataset")


# -- Helpers --


def _error(ctx: ExecutionContext, message: str) -> object:
    """Yield-safe error via set_progress (raising breaks generator streams)."""
    label = f"Error: {message}"
    if ctx.delegated:
        ctx.set_progress(progress=0, label=label)
        return None
    return ctx.trigger("set_progress", {"progress": 0, "label": label})


def _parse_batch_responses(
    inf: _InferenceContext,
    batch_ids: list[str],
    responses: list[str | Exception],
    batch_widths: list[int | None],
    batch_heights: list[int | None],
) -> tuple[dict[str, object], dict[str, str], int]:
    """Parse inference responses into results and errors.

    Returns:
        Tuple of (results dict, errors dict, error count).
    """
    results: dict[str, object] = {}
    errors: dict[str, str] = {}
    error_count = 0
    for sid, resp, img_w, img_h in zip(batch_ids, responses, batch_widths, batch_heights):
        if isinstance(resp, Exception):
            errors[sid] = f"{type(resp).__name__}: {resp}"
            error_count += 1
            continue
        try:
            label = inf.task.parse_response(resp, image_width=img_w, image_height=img_h)
            if inf.log_metadata and inf.metadata:
                label.model_name = inf.metadata["model_name"]
                label.prompt = inf.metadata["prompt"]
                label.infer_cfg = inf.metadata["infer_cfg"]
            results[sid] = label
        except Exception as e:
            errors[sid] = f"{type(e).__name__}: {e}"
            error_count += 1
    return results, errors, error_count


def _build_metadata(
    params: dict[str, object],
    engine: VLLMEngine,
    task: TaskConfig,
    batch_size: int,
    image_mode: str,
) -> dict[str, object]:
    """Build run metadata dict for per-label and dataset-level logging."""
    full_prompt = ""
    if task.system_prompt:
        full_prompt += f"[system] {task.system_prompt}\n"
    full_prompt += f"[user] {task.prompt}"
    return {
        "model_name": params["model"],
        "prompt": full_prompt,
        "infer_cfg": {
            "temperature": engine.temperature,
            "max_tokens": engine.max_tokens,
            "top_p": engine.top_p,
            "batch_size": batch_size,
            "coordinate_format": task.coordinate_format,
            "box_format": task.box_format,
            "image_mode": image_mode,
            "max_concurrent": engine.max_concurrent,
        },
    }


def _clear_stale_errors(
    dataset: fo.Dataset,
    field_name: str,
    ids: list[str],
    overwrite: bool,
) -> None:
    """Clear stale error fields when overwriting previous results."""
    if not overwrite:
        return
    error_field = f"{field_name}_error"
    schema = dataset.get_field_schema(flat=True)
    if error_field in schema:
        dataset.set_values(error_field, {sid: None for sid in ids}, key_field="id")


def _resolve_config(ctx: ExecutionContext) -> dict[str, object]:
    """Merge dataset config > global config > _DEFAULTS."""
    merged = dict(_DEFAULTS)
    for cfg in (get_global_config(), ctx.dataset.info.get("_vllm_config") or {}):
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _create_engine(
    params: dict[str, object],
    secrets: dict[str, str | None],
) -> tuple[VLLMEngine, str, str]:
    """Build and validate a VLLMEngine from operator params and secrets.

    Returns (engine, base_url, api_key) — base_url/api_key needed for
    dataset.info persistence.
    """
    api_key = params.get("api_key") or secrets.get("FIFTYONE_VLLM_API_KEY") or "EMPTY"
    base_url = params.get("base_url") or secrets.get("FIFTYONE_VLLM_BASE_URL") or "http://localhost:8000/v1"

    engine = VLLMEngine(
        EngineConfig(
            model=params["model"],
            base_url=base_url,
            api_key=api_key,
            max_concurrent=params.get("max_concurrent", _DEFAULTS["max_concurrent"]),
            temperature=params.get("temperature"),
            max_tokens=params.get("max_tokens", _DEFAULTS["max_tokens"]),
            top_p=params.get("top_p", _DEFAULTS["top_p"]),
            seed=params.get("seed"),
        )
    )
    engine.validate_connection()

    return engine, base_url, api_key


def _create_task(params: dict[str, object]) -> TaskConfig:
    """Build a TaskConfig from operator params."""
    classes = normalize_classes(params.get("classes"))
    detection = DetectionConfig(
        coordinate_format=params.get("coordinate_format", _DEFAULTS["coordinate_format"]),
        box_format=params.get("box_format", _DEFAULTS["box_format"]),
    )

    return TaskConfig(
        task=params["task"],
        prompt=params.get("prompt_override") or params.get("prompt"),
        system_prompt=params.get("system_prompt"),
        classes=classes,
        detection=detection,
        question=params.get("question", ""),
    )


def _resolve_field_name(
    dataset: fo.Dataset,
    task_name: str,
    overwrite: bool = False,
) -> str:
    """Resolve the output field name for a task run.

    Produces: vllm_infer_task_name, vllm_infer_task_name1, ...

    Args:
        dataset: FiftyOne dataset to check for existing fields.
        task_name: task identifier used in the field name.
        overwrite: if True, reuse the highest existing field name
            instead of incrementing.
    """
    schema = dataset.get_field_schema(flat=True)
    base = f"vllm_infer_{task_name}"

    if base not in schema:
        return base

    n = 1
    while f"{base}{n}" in schema:
        n += 1

    if overwrite:
        return f"{base}{n - 1}" if n > 1 else base

    return f"{base}{n}"


def _write_batch_results(
    dataset: fo.Dataset,
    field_name: str,
    results: dict[str, object],
    errors: dict[str, str],
) -> None:
    """Write a batch of results and errors as flat sample fields.

    Args:
        dataset: FiftyOne dataset to write to.
        field_name: base field name for results.
        results: dict mapping sample ID to FiftyOne label.
        errors: dict mapping sample ID to error string.
    """
    if results:
        dataset.set_values(
            field_name,
            results,
            key_field="id",
            dynamic=True,
        )
    if errors:
        dataset.set_values(
            f"{field_name}_error",
            errors,
            key_field="id",
            dynamic=True,
        )


# -- UI helper functions --


def _json_config_inputs(ctx: ExecutionContext, inputs: types.Object) -> None:
    """Add JSON paste mode UI fields and validation to the input form."""
    inputs.str(
        "config_json",
        label="Paste JSON Config",
        required=True,
        description="Paste a config exported from a previous run",
        view=types.CodeView(language="json"),
    )
    inputs.bool(
        "show_params",
        label="Show accepted parameters",
        default=False,
        view=types.SwitchView(),
    )
    if ctx.params.get("show_params"):
        inputs.md(
            "**Server:** `model`, `base_url`\n\n"
            "**Task:** `task`, `classes`, `question`, `prompt`, "
            "`system_prompt`, `prompt_override`\n\n"
            "**Advanced:** `temperature`, `max_tokens`, `top_p`, "
            "`seed`, `batch_size`, `max_concurrent`, `max_workers`, "
            "`image_mode`, `coordinate_format`, `box_format`",
            name="params_ref",
        )
    raw = ctx.params.get("config_json")
    if raw:
        _validate_json_config(inputs, raw)


def _validate_json_config(inputs: types.Object, raw: str) -> None:
    """Parse and validate a JSON config string, adding status views to the form."""
    cfg, err = parse_config_json(raw)
    if err:
        inputs.view("json_err", types.Error(label=err))
        return
    missing = [k for k in ("model", "task") if not cfg.get(k)]
    if cfg.get("task") == "vqa" and not cfg.get("question"):
        missing.append("question")
    if missing:
        inputs.view(
            "json_warn",
            types.Warning(label="Missing required: " + ", ".join(missing)),
        )
    else:
        inputs.view(
            "json_ok",
            types.Notice(label=f"Valid: {cfg['task']} task with {cfg['model']}"),
        )


def _model_selector(inputs: types.Object, stored: dict[str, object]) -> None:
    """Add model selection field to the input form."""
    inputs.str(
        "model",
        label="Model",
        required=True,
        default=stored.get("model", ""),
        description=("HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)"),
    )


def _server_settings(inputs: types.Object, stored: dict[str, object]) -> None:
    """Add server URL and API key fields to the input form."""
    inputs.view(
        "server_header",
        types.Header(label="Server Settings", divider=True),
    )
    inputs.str(
        "base_url",
        label="Server URL",
        required=True,
        default=stored.get("base_url"),
        description="vLLM OpenAI-compatible API endpoint",
    )
    inputs.str(
        "api_key",
        label="API Key",
        default=stored.get("api_key"),
        description="API key for authentication (use 'EMPTY' for no auth)",
    )


def _task_selector(
    ctx: ExecutionContext,
    inputs: types.Object,
    stored: dict[str, object],
) -> str | None:
    """Add task dropdown to the input form.

    Returns:
        Selected task name, or None if not yet chosen.
    """
    task_dropdown = types.Dropdown(label="Task")
    task_dropdown.add_choice(
        "caption",
        label="Caption",
        description="Generate a text description of the image",
    )
    task_dropdown.add_choice(
        "classify",
        label="Classify",
        description="Assign a single class label (constrained output)",
    )
    task_dropdown.add_choice(
        "tag",
        label="Tag",
        description="Assign multiple labels (constrained JSON output)",
    )
    task_dropdown.add_choice(
        "detect",
        label="Detect",
        description=("Detect objects with bounding boxes (constrained JSON output)"),
    )
    task_dropdown.add_choice(
        "vqa",
        label="VQA",
        description="Answer a question about the image",
    )
    task_dropdown.add_choice(
        "ocr",
        label="OCR",
        description="Extract text visible in the image",
    )
    inputs.enum(
        "task",
        task_dropdown.values(),
        required=True,
        default=stored.get("task"),
        label="Task",
        view=task_dropdown,
    )
    return ctx.params.get("task", None)


def _task_settings(
    inputs: types.Object,
    task: str | None,
    stored: dict[str, object],
) -> None:
    """Add task-specific settings (classes, question, prompt) to the form."""
    if task is None:
        return

    inputs.view(
        "task_header",
        types.Header(label="Task Settings", divider=True),
    )

    if task in ("classify", "tag", "detect"):
        stored_classes = stored.get("classes") or ""
        if isinstance(stored_classes, list):
            stored_classes = ", ".join(stored_classes)
        inputs.str(
            "classes",
            label="Classes",
            required=False,
            default=stored_classes,
            description=("Comma-separated class names (leave empty for open-ended)"),
        )

    if task == "vqa":
        inputs.str(
            "question",
            label="Question",
            required=True,
            default=stored.get("question", ""),
            description="Question to ask about each image",
        )

    inputs.str(
        "prompt_override",
        label="Prompt Override",
        required=False,
        default=stored.get("prompt_override", ""),
        description="Override the default prompt for this task",
    )


def _output_settings(
    ctx: ExecutionContext,
    inputs: types.Object,
    task: str | None,
) -> None:
    """Add output field and metadata logging settings to the form."""
    if not task or not ctx.dataset:
        return

    inputs.view(
        "output_header",
        types.Header(label="Output Settings", divider=True),
    )

    schema = ctx.dataset.get_field_schema(flat=True)
    base_field = f"vllm_infer_{task}"
    has_existing = base_field in schema

    if has_existing:
        inputs.bool(
            "overwrite_last",
            label="Overwrite last result",
            default=False,
            view=types.SwitchView(),
        )
        overwrite = ctx.params.get("overwrite_last", False)
        resolved = _resolve_field_name(ctx.dataset, task, overwrite)
        prefix = "Overwriting" if overwrite else "Writing to"
        inputs.view(
            "field_info",
            types.Notice(label=f"{prefix}: {resolved}"),
        )
    else:
        inputs.view(
            "field_info",
            types.Notice(label=f"Writing to: {base_field}"),
        )

    inputs.bool(
        "log_metadata",
        label="Log run metadata",
        default=False,
        view=types.SwitchView(),
        description=("Store model name, prompt, and inference config on each result label and in dataset info"),
    )


def _advanced_settings(
    ctx: ExecutionContext,
    inputs: types.Object,
    stored: dict[str, object],
) -> None:
    """Add collapsible advanced settings (temperature, batch size, etc.)."""
    inputs.view(
        "adv_header",
        types.Header(label="Advanced Settings", divider=True),
    )
    inputs.bool(
        "show_advanced",
        label="Show advanced settings",
        default=False,
        view=types.SwitchView(),
    )
    if not ctx.params.get("show_advanced", False):
        return

    inputs.float(
        "temperature",
        label="Temperature",
        default=stored.get("temperature"),
        min=0.0,
        max=2.0,
        description=("Sampling temperature (leave empty for task-specific default)"),
    )
    inputs.int(
        "max_tokens",
        label="Max Tokens",
        default=stored.get("max_tokens"),
        min=1,
        max=4096,
        description="Maximum tokens to generate per sample",
    )
    inputs.float(
        "top_p",
        label="Top P",
        default=stored.get("top_p"),
        min=0.0,
        max=1.0,
    )
    inputs.int(
        "seed",
        label="Seed",
        default=stored.get("seed"),
        description="Random seed for reproducible results (leave empty for non-deterministic)",
    )
    inputs.int(
        "batch_size",
        label="Batch Size",
        default=stored.get("batch_size"),
        min=1,
        max=512,
        description="Number of samples per inference batch",
    )
    inputs.int(
        "max_concurrent",
        label="Max Concurrent Requests",
        default=stored.get("max_concurrent"),
        min=1,
        max=256,
        description="Maximum parallel HTTP requests to vLLM server",
    )
    inputs.int(
        "max_workers",
        label="Image Loading Workers",
        default=stored.get("max_workers"),
        min=1,
        max=32,
        description=("Thread pool size for parallel image loading/encoding"),
    )

    image_dropdown = types.Dropdown()
    image_dropdown.add_choice(
        "auto",
        label="Auto (Recommended)",
        description="URLs passed through, local files base64-encoded",
    )
    image_dropdown.add_choice(
        "filepath",
        label="File Path",
        description=("file:// paths (local vLLM server with --allowed-local-media-path only)"),
    )
    inputs.enum(
        "image_mode",
        image_dropdown.values(),
        default=stored.get("image_mode"),
        label="Image Mode",
        view=image_dropdown,
    )

    task = ctx.params.get("task")
    if task == "detect":
        coord_dropdown = types.Dropdown()
        coord_dropdown.add_choice("normalized_1000", label="0-1000 (Qwen default)")
        coord_dropdown.add_choice("normalized_1", label="0-1 (relative)")
        coord_dropdown.add_choice("pixel", label="Pixel coordinates")
        inputs.enum(
            "coordinate_format",
            coord_dropdown.values(),
            default=stored.get("coordinate_format"),
            label="Coordinate Format",
            view=coord_dropdown,
            description=("Bounding box coordinate convention used by the model"),
        )

        box_dropdown = types.Dropdown()
        box_dropdown.add_choice("xyxy", label="xyxy — corners")
        box_dropdown.add_choice("xywh", label="xywh — origin + size")
        box_dropdown.add_choice("cxcywh", label="cxcywh — center + size")
        inputs.enum(
            "box_format",
            box_dropdown.values(),
            default=stored.get("box_format"),
            label="Box Format",
            view=box_dropdown,
            description="Bounding box format produced by the model",
        )
