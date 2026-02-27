"""FiftyOne operator for vLLM inference: UI, batching, progress, and result
storage."""

import fiftyone.operators as foo
from fiftyone.operators import types

from .engine import VLLMEngine
from .tasks import TaskConfig
from .utils import build_image_contents

_DEFAULTS = {
    "batch_size": 8,
    "max_tokens": 512,
    "top_p": 1.0,
    "max_concurrent": 16,
    "max_workers": 4,
    "image_mode": "auto",
    "coordinate_format": "normalized_1000",
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

        _model_selector(ctx, inputs)
        _server_settings(ctx, inputs)
        task = _task_selector(ctx, inputs)
        _task_settings(ctx, inputs, task)
        _output_settings(ctx, inputs, task)
        _advanced_settings(ctx, inputs)
        inputs.view_target(ctx)

        return types.Property(inputs, view=types.View(label="vLLM Inference"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        params = ctx.params

        engine, base_url, api_key = _create_engine(params, ctx.secrets)
        task = _create_task(params)

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

        structured_outputs = task.get_structured_outputs()

        # 5a. Resolve output field name (vllm_infer_caption, ...)
        field_name = _resolve_field_name(
            ctx.dataset,
            params["task"],
            params.get("overwrite_last", False),
        )

        # 5b. Build metadata for optional per-label logging
        log_metadata = params.get("log_metadata", False)
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

        # 5c. Collect image dimensions for pixel coordinate normalization
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
                structured_outputs=structured_outputs,
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

        ctx.dataset.info["_vllm_config"] = {
            "model": params["model"],
            "base_url": base_url,
            "api_key": api_key,
        }
        ctx.dataset.save()

        # 8. Reload dataset in App
        if not ctx.delegated:
            yield ctx.trigger("reload_dataset")

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("summary", label="Summary")
        return types.Property(outputs, view=types.View(label="Complete"))


# -- Helpers --


def _stored(ctx):
    """Read persisted settings from dataset info."""
    if not ctx.dataset:
        return {}
    return ctx.dataset.info.get("_vllm_config") or {}


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
    )
    engine.validate_connection()

    return engine, base_url, api_key


def _create_task(params):
    """Build a TaskConfig from operator params."""
    classes = None
    raw_classes = params.get("classes")
    if raw_classes:
        classes = [c.strip() for c in raw_classes.split(",")]

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
    """Write a batch of results and errors as flat sample fields."""
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


def _model_selector(ctx, inputs):
    stored = _stored(ctx)
    inputs.str(
        "model",
        label="Model",
        required=True,
        default=stored.get("model", ""),
        description=("HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)"),
    )


def _server_settings(ctx, inputs):
    stored = _stored(ctx)
    inputs.view(
        "server_header",
        types.Header(label="Server Settings", divider=True),
    )
    inputs.str(
        "base_url",
        label="Server URL",
        required=True,
        default=stored.get("base_url", "http://localhost:8000/v1"),
        description="vLLM OpenAI-compatible API endpoint",
    )
    inputs.str(
        "api_key",
        label="API Key",
        default=stored.get("api_key", "EMPTY"),
        description="API key for authentication (use 'EMPTY' for no auth)",
    )


def _task_selector(ctx, inputs):
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
    task_dropdown.add_choice(
        "custom",
        label="Custom",
        description="Custom prompt with free-form response",
    )
    inputs.enum(
        "task",
        task_dropdown.values(),
        required=True,
        label="Task",
        view=task_dropdown,
    )
    return ctx.params.get("task", None)


def _task_settings(ctx, inputs, task):
    if task is None:
        return

    inputs.view(
        "task_header",
        types.Header(label="Task Settings", divider=True),
    )

    if task in ("classify", "tag"):
        inputs.str(
            "classes",
            label="Classes",
            required=True,
            description=("Comma-separated class names (e.g., cat, dog, bird)"),
        )
    elif task == "detect":
        inputs.str(
            "classes",
            label="Classes",
            required=False,
            description=(
                "Optional: comma-separated classes to detect"
                " (leave empty for open detection)"
            ),
        )

    if task == "vqa":
        inputs.str(
            "question",
            label="Question",
            required=True,
            description="Question to ask about each image",
        )

    if task == "custom":
        inputs.str(
            "prompt",
            label="Prompt",
            required=True,
            description="Prompt to send with each image",
        )
        inputs.str(
            "system_prompt",
            label="System Prompt",
            required=False,
            description="Optional system prompt for context",
        )

    if task != "custom":
        inputs.str(
            "prompt_override",
            label="Prompt Override",
            required=False,
            description="Override the default prompt for this task",
        )


def _output_settings(ctx, inputs, task):
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


def _advanced_settings(ctx, inputs):
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
        default=None,
        min=0.0,
        max=2.0,
        description=("Sampling temperature (leave empty for task-specific default)"),
    )
    inputs.int(
        "max_tokens",
        label="Max Tokens",
        default=_DEFAULTS["max_tokens"],
        min=1,
        max=4096,
        description="Maximum tokens to generate per sample",
    )
    inputs.float(
        "top_p",
        label="Top P",
        default=_DEFAULTS["top_p"],
        min=0.0,
        max=1.0,
    )
    inputs.int(
        "batch_size",
        label="Batch Size",
        default=_DEFAULTS["batch_size"],
        min=1,
        max=512,
        description="Number of samples per inference batch",
    )

    inputs.int(
        "max_concurrent",
        label="Max Concurrent Requests",
        default=_DEFAULTS["max_concurrent"],
        min=1,
        max=256,
        description="Maximum parallel HTTP requests to vLLM server",
    )
    inputs.int(
        "max_workers",
        label="Image Loading Workers",
        default=_DEFAULTS["max_workers"],
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
        default=_DEFAULTS["image_mode"],
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
            default=_DEFAULTS["coordinate_format"],
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
            default=_DEFAULTS["box_format"],
            label="Box Format",
            view=box_dropdown,
            description="Bounding box format produced by the model",
        )
