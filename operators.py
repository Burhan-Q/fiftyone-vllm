"""FiftyOne operator for vLLM inference: UI, batching, progress, and result
storage."""

import json

import fiftyone.operators as foo
from fiftyone.operators import types

from .engine import VLLMEngine
from .tasks import TaskConfig
from .utils import (
    build_image_contents,
    clear_global_config,
    get_global_config,
    normalize_classes,
    parse_config_json,
    pick_params,
    save_global_config,
)

_DEFAULTS = {
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


class VLLMInference(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_vllm_inference",
            label="Run vLLM Inference",
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not ctx.dataset:
            inputs.view("error", types.Error(label="No dataset loaded"))
            return types.Property(inputs)

        stored = _resolve_config(ctx)

        # Config mode selector
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
                cfg, err = parse_config_json(raw)
                if err:
                    inputs.view("json_err", types.Error(label=err))
                else:
                    missing = [k for k in ("model", "task") if not cfg.get(k)]
                    if cfg.get("task") == "vqa" and not cfg.get("question"):
                        missing.append("question")
                    if missing:
                        inputs.view(
                            "json_warn",
                            types.Warning(
                                label="Missing required: " + ", ".join(missing)
                            ),
                        )
                    else:
                        inputs.view(
                            "json_ok",
                            types.Notice(
                                label=f"Valid: {cfg['task']} task with {cfg['model']}"
                            ),
                        )
        elif config_mode == "reset":
            inputs.view(
                "reset_notice",
                types.Notice(
                    label="All stored settings (global and dataset) will be"
                    " cleared and defaults restored."
                ),
            )
        else:
            _model_selector(ctx, inputs, stored)
            _server_settings(ctx, inputs, stored)
            task = _task_selector(ctx, inputs, stored)
            _task_settings(ctx, inputs, task, stored)
            _output_settings(ctx, inputs, task)
            _advanced_settings(ctx, inputs, stored)

        if config_mode != "reset":
            inputs.view_target(ctx)

        return types.Property(inputs, view=types.View(label="vLLM Inference"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", None)

    def execute(self, ctx):
        params = ctx.params
        config_mode = params.get("config_mode", "manual")

        # Handle reset mode — clear all stored configs
        if config_mode == "reset":
            clear_global_config()
            ctx.dataset.info.pop("_vllm_config", None)
            ctx.dataset.save()
            if not ctx.delegated:
                yield ctx.trigger("reload_dataset")
            return

        # Handle JSON paste mode — merge imported config into params
        if config_mode == "json":
            raw = params.get("config_json") or ""
            cfg, err = parse_config_json(raw)
            if err:
                yield _error(ctx, f"Config import failed: {err}")
                return
            params.update(cfg)
            if not params.get("model") or not params.get("task"):
                yield _error(ctx, "Config missing required 'model' or 'task'")
                return

        try:
            engine, base_url, api_key = _create_engine(params, ctx.secrets)
            task = _create_task(params)
        except Exception as e:
            yield _error(ctx, str(e))
            return

        if params.get("temperature") is None:
            engine.temperature = task.default_temperature

        batch_size = params.get("batch_size", _DEFAULTS["batch_size"])
        image_mode = params.get("image_mode", _DEFAULTS["image_mode"])

        # 5. Get target samples and resolve output field
        view = ctx.target_view()
        ids = view.values("id")
        filepaths = view.values("filepath")
        total = len(ids)
        max_workers = params.get("max_workers", _DEFAULTS["max_workers"])

        response_model = task.get_response_model()

        # 5a. Resolve output field name (vllm_infer_caption, ...)
        field_name = _resolve_field_name(
            ctx.dataset,
            params["task"],
            params.get("overwrite_last", False),
        )

        # 5b. Build metadata for optional per-label logging
        log_metadata = params.get("log_metadata", False)
        if log_metadata:
            full_prompt = ""
            if task.system_prompt:
                full_prompt += f"[system] {task.system_prompt}\n"
            full_prompt += f"[user] {task.prompt}"

            infer_cfg = {
                "temperature": engine.temperature,
                "max_tokens": engine.max_tokens,
                "top_p": engine.top_p,
                "batch_size": batch_size,
                "coordinate_format": task.coordinate_format,
                "box_format": task.box_format,
                "image_mode": image_mode,
                "max_concurrent": engine.max_concurrent,
            }

        # 5c. Clear stale error fields when overwriting
        if params.get("overwrite_last", False):
            error_field = f"{field_name}_error"
            schema = ctx.dataset.get_field_schema(flat=True)
            if error_field in schema:
                ctx.dataset.set_values(
                    error_field,
                    {sid: None for sid in ids},
                    key_field="id",
                )

        # 5d. Collect image dimensions for pixel coordinate normalization
        need_dims = task.task == "detect" and task.coordinate_format == "pixel"
        if need_dims:
            view.compute_metadata()
            widths = view.values("metadata.width")
            heights = view.values("metadata.height")
        else:
            widths = [None] * total
            heights = [None] * total

        # 6. Process in batches
        processed = 0
        total_errors = 0

        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_paths = filepaths[i : i + batch_size]
            batch_widths = widths[i : i + batch_size]
            batch_heights = heights[i : i + batch_size]

            # 6a. Parallel image content construction
            image_contents = build_image_contents(
                batch_paths,
                image_mode=image_mode,
                max_workers=max_workers,
            )

            # 6b. Build messages for each image
            batch_messages = [task.build_messages(img) for img in image_contents]

            # 6c. Batch inference with structured output
            responses = engine.infer_batch(
                batch_messages,
                response_model=response_model,
            )

            # 6d. Parse responses with per-sample error handling
            results = {}
            errors = {}
            for sid, resp, img_w, img_h in zip(
                batch_ids, responses, batch_widths, batch_heights
            ):
                if isinstance(resp, Exception):
                    errors[sid] = f"{type(resp).__name__}: {resp}"
                    total_errors += 1
                    continue
                try:
                    label = task.parse_response(
                        resp, image_width=img_w, image_height=img_h
                    )
                    if log_metadata:
                        label.model_name = params["model"]
                        label.prompt = full_prompt
                        label.infer_cfg = infer_cfg
                    results[sid] = label
                except Exception as e:
                    errors[sid] = f"{type(e).__name__}: {e}"
                    total_errors += 1

            # 6e. Bulk-write results and errors as flat fields
            _write_batch_results(
                ctx.dataset,
                field_name,
                results,
                errors,
            )

            # 6f. Progress
            processed += len(batch_ids)
            label = f"{processed}/{total} samples"
            if total_errors:
                label += f" ({total_errors} errors)"

            if ctx.delegated:
                ctx.set_progress(progress=processed / total, label=label)
            else:
                yield ctx.trigger(
                    "set_progress",
                    dict(progress=processed / total, label=label),
                )

        # 7. Persist run metadata and settings
        if log_metadata:
            runs = ctx.dataset.info.get("vllm_runs", {})
            runs[field_name] = {
                "model_name": params["model"],
                "prompt": full_prompt,
                "infer_cfg": infer_cfg,
            }
            ctx.dataset.info["vllm_runs"] = runs

        # Persist full config to both tiers
        params["base_url"] = base_url
        params["api_key"] = api_key
        save_global_config(params)
        ctx.dataset.info["_vllm_config"] = pick_params(params)
        ctx.dataset.save()

        # 8. Notify / reload
        if ctx.delegated:
            # Signal the store so CheckVLLMStatus (running in the App
            # process) is woken up via the change-stream notification.
            store = ctx.store("vllm_status")
            store.set("done", True)
        else:
            yield ctx.trigger("reload_dataset")

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("summary", label="Summary")

        if ctx.params.get("config_mode") != "reset":
            cfg_json = json.dumps(
                pick_params(ctx.params, exclude=("api_key",)), indent=2
            )
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
    def config(self):
        return foo.OperatorConfig(
            name="check_vllm_status",
            label="Check vLLM Status",
            on_dataset_open=True,
            execute_as_generator=True,
            unlisted=True,
        )

    async def execute(self, ctx):
        import asyncio

        from fiftyone.operators.store.notification_service import (
            default_notification_service,
        )

        if not ctx.dataset:
            return

        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        def _on_change(_message):
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


# -- Helpers --


def _error(ctx, message):
    """Yield-safe error via set_progress (raising breaks generator streams)."""
    label = f"Error: {message}"
    if ctx.delegated:
        ctx.set_progress(progress=0, label=label)
        return None
    return ctx.trigger("set_progress", {"progress": 0, "label": label})


def _resolve_config(ctx):
    """Merge dataset config > global config > _DEFAULTS."""
    merged = dict(_DEFAULTS)
    for cfg in (get_global_config(), ctx.dataset.info.get("_vllm_config") or {}):
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _create_engine(params, secrets):
    """Build and validate a VLLMEngine from operator params and secrets.

    Returns (engine, base_url, api_key) — base_url/api_key needed for
    dataset.info persistence.
    """
    api_key = (
        params.get("api_key") or secrets.get("FIFTYONE_VLLM_API_KEY", None) or "EMPTY"
    )
    base_url = (
        params.get("base_url")
        or secrets.get("FIFTYONE_VLLM_BASE_URL", None)
        or "http://localhost:8000/v1"
    )

    engine = VLLMEngine(
        model=params["model"],
        base_url=base_url,
        api_key=api_key,
        max_concurrent=params.get("max_concurrent", _DEFAULTS["max_concurrent"]),
        temperature=params.get("temperature", None),
        max_tokens=params.get("max_tokens", _DEFAULTS["max_tokens"]),
        top_p=params.get("top_p", _DEFAULTS["top_p"]),
        seed=params.get("seed", None),
    )
    engine.validate_connection()

    return engine, base_url, api_key


def _create_task(params):
    """Build a TaskConfig from operator params."""
    classes = normalize_classes(params.get("classes"))

    return TaskConfig(
        task=params["task"],
        prompt=params.get("prompt_override") or params.get("prompt"),
        system_prompt=params.get("system_prompt"),
        classes=classes,
        coordinate_format=params.get(
            "coordinate_format", _DEFAULTS["coordinate_format"]
        ),
        box_format=params.get("box_format", _DEFAULTS["box_format"]),
        question=params.get("question", ""),
    )


def _resolve_field_name(dataset, task_name, overwrite=False):
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


def _write_batch_results(dataset, field_name, results, errors):
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


def _model_selector(ctx, inputs, stored):
    """Add model selection field to the input form.

    Args:
        ctx: operator execution context.
        inputs: form object to populate.
        stored: resolved config dict with defaults merged.
    """
    inputs.str(
        "model",
        label="Model",
        required=True,
        default=stored.get("model", ""),
        description=("HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)"),
    )


def _server_settings(ctx, inputs, stored):
    """Add server URL and API key fields to the input form.

    Args:
        ctx: operator execution context.
        inputs: form object to populate.
        stored: resolved config dict with defaults merged.
    """
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


def _task_selector(ctx, inputs, stored):
    """Add task dropdown to the input form.

    Args:
        ctx: operator execution context.
        inputs: form object to populate.
        stored: resolved config dict with defaults merged.

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


def _task_settings(ctx, inputs, task, stored):
    """Add task-specific settings (classes, question, prompt) to the form.

    Args:
        ctx: operator execution context.
        inputs: form object to populate.
        task: selected task name (e.g. "classify", "vqa"), or None.
        stored: resolved config dict with defaults merged.
    """
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


def _output_settings(ctx, inputs, task):
    """Add output field and metadata logging settings to the form.

    Args:
        ctx: operator execution context.
        inputs: form object to populate.
        task: selected task name.
    """
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
        description=(
            "Store model name, prompt, and inference config"
            " on each result label and in dataset info"
        ),
    )


def _advanced_settings(ctx, inputs, stored):
    """Add collapsible advanced settings (temperature, batch size, etc.).

    Args:
        ctx: operator execution context.
        inputs: form object to populate.
        stored: resolved config dict with defaults merged.
    """
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
        description=(
            "file:// paths (local vLLM server with --allowed-local-media-path only)"
        ),
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
