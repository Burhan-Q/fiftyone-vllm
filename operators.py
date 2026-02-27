"""FiftyOne operator for vLLM inference: UI, batching, progress, and result
storage."""

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .engine import VLLMEngine
from .tasks import TaskConfig
from .utils import build_image_contents


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
        _field_conflict_check(ctx, inputs, task)
        _advanced_settings(ctx, inputs)
        inputs.view_target(ctx)

        return types.Property(
            inputs, view=types.View(label="vLLM Inference")
        )

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        params = ctx.params

        # 1. Resolve secrets with precedence:
        #    UI param > FiftyOne secret > env > default
        api_key = (
            params.get("api_key")
            or ctx.secrets.get("FIFTYONE_VLLM_API_KEY", None)
            or "EMPTY"
        )
        base_url = (
            params.get("base_url")
            or ctx.secrets.get("FIFTYONE_VLLM_BASE_URL", None)
            or "http://localhost:8000/v1"
        )

        # 2. Build engine
        engine = VLLMEngine(
            model=params["model"],
            base_url=base_url,
            api_key=api_key,
            max_concurrent=params.get("max_concurrent", 64),
            temperature=params.get("temperature", None),
            max_tokens=params.get("max_tokens", 512),
            top_p=params.get("top_p", 1.0),
        )

        # 3. Validate connection
        engine.validate_connection()

        # 4. Build task config
        classes = None
        raw_classes = params.get("classes")
        if raw_classes:
            classes = [c.strip() for c in raw_classes.split(",")]

        task = TaskConfig(
            task=params["task"],
            prompt=params.get("prompt_override") or params.get("prompt"),
            system_prompt=params.get("system_prompt"),
            classes=classes,
            coordinate_format=params.get(
                "coordinate_format", "normalized_1000"
            ),
            question=params.get("question", ""),
        )

        # Apply task-specific temperature default if user didn't override
        if params.get("temperature") is None:
            engine.temperature = task.default_temperature

        # 5. Get target samples
        view = ctx.target_view()
        ids = view.values("id")
        filepaths = view.values("filepath")
        total = len(ids)
        batch_size = params.get("batch_size", 32)
        output_field = params["output_field"]
        max_workers = params.get("max_workers", 8)

        structured_outputs = task.get_structured_outputs()

        # 6. Process in batches
        processed = 0
        total_errors = 0

        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_paths = filepaths[i : i + batch_size]

            # 6a. Parallel image content construction
            image_contents = build_image_contents(
                batch_paths,
                image_mode=params.get("image_mode", "auto"),
                base_url=base_url,
                max_workers=max_workers,
            )

            # 6b. Build messages for each image
            batch_messages = [
                task.build_messages(img) for img in image_contents
            ]

            # 6c. Batch inference with structured output
            responses = engine.infer_batch(
                batch_messages,
                structured_outputs=structured_outputs,
            )

            # 6d. Parse responses with per-sample error handling
            results = {}
            errors = {}
            for sid, resp in zip(batch_ids, responses):
                if isinstance(resp, Exception):
                    errors[sid] = f"{type(resp).__name__}: {resp}"
                    total_errors += 1
                    continue
                try:
                    results[sid] = task.parse_response(resp)
                except Exception as e:
                    errors[sid] = f"{type(e).__name__}: {e}"
                    total_errors += 1

            # 6e. Bulk-write results and errors
            if results:
                ctx.dataset.set_values(
                    output_field, results, key_field="id"
                )
            if errors:
                ctx.dataset.set_values(
                    f"{output_field}_error", errors, key_field="id"
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

        # 7. Reload dataset in App
        if not ctx.delegated:
            yield ctx.trigger("reload_dataset")

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("summary", label="Summary")
        return types.Property(
            outputs, view=types.View(label="Complete")
        )


# -- UI helper functions --


def _model_selector(ctx, inputs):
    inputs.str(
        "model",
        label="Model",
        required=True,
        description=(
            "HuggingFace model ID"
            " (e.g., Qwen/Qwen2.5-VL-7B-Instruct)"
        ),
    )


def _server_settings(ctx, inputs):
    inputs.view(
        "server_header",
        types.Header(label="Server Settings", divider=True),
    )
    inputs.str(
        "base_url",
        label="Server URL",
        required=True,
        default="http://localhost:8000/v1",
        description="vLLM OpenAI-compatible API endpoint",
    )
    inputs.str(
        "api_key",
        label="API Key",
        default="EMPTY",
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
        description=(
            "Detect objects with bounding boxes (constrained JSON output)"
        ),
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
            description=(
                "Comma-separated class names (e.g., cat, dog, bird)"
            ),
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
    defaults = {t: v["default_field"] for t, v in TaskConfig.TASKS.items()}
    inputs.str(
        "output_field",
        label="Output Field",
        required=True,
        default=defaults.get(task, "vlm_output"),
        description="Field name to store results on each sample",
    )


def _field_conflict_check(ctx, inputs, task):
    """Check for field type conflicts and warn in the UI."""
    output_field = ctx.params.get("output_field")
    if not output_field or not ctx.dataset:
        return

    existing = ctx.dataset.get_field(output_field)
    if existing is None:
        return

    expected_types = {
        "caption": fo.StringField,
        "classify": fo.EmbeddedDocumentField,
        "tag": fo.EmbeddedDocumentField,
        "detect": fo.EmbeddedDocumentField,
        "vqa": fo.StringField,
        "ocr": fo.StringField,
        "custom": fo.StringField,
    }
    expected = expected_types.get(task)
    if expected and not isinstance(existing, expected):
        inputs.view(
            "field_warning",
            types.Warning(
                label=(
                    f"Field '{output_field}' already exists with type"
                    f" {type(existing).__name__}. This task writes"
                    f" {expected.__name__}. Choose a different field name"
                    " to avoid conflicts."
                )
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
        description=(
            "Sampling temperature"
            " (leave empty for task-specific default)"
        ),
    )
    inputs.int(
        "max_tokens",
        label="Max Tokens",
        default=512,
        min=1,
        max=4096,
        description="Maximum tokens to generate per sample",
    )
    inputs.float(
        "top_p",
        label="Top P",
        default=1.0,
        min=0.0,
        max=1.0,
    )
    inputs.int(
        "batch_size",
        label="Batch Size",
        default=32,
        min=1,
        max=512,
        description="Number of samples per inference batch",
    )

    inputs.int(
        "max_concurrent",
        label="Max Concurrent Requests",
        default=64,
        min=1,
        max=256,
        description="Maximum parallel HTTP requests to vLLM server",
    )
    inputs.int(
        "max_workers",
        label="Image Loading Workers",
        default=8,
        min=1,
        max=32,
        description=(
            "Thread pool size for parallel image loading/encoding"
        ),
    )

    image_dropdown = types.Dropdown()
    image_dropdown.add_choice(
        "auto",
        label="Auto",
        description="Best option based on server location",
    )
    image_dropdown.add_choice(
        "base64",
        label="Base64",
        description="Base64-encoded (works everywhere)",
    )
    image_dropdown.add_choice(
        "filepath",
        label="File Path",
        description=(
            "file:// paths (local servers with"
            " --allowed-local-media-path)"
        ),
    )
    inputs.enum(
        "image_mode",
        image_dropdown.values(),
        default="auto",
        label="Image Mode",
        view=image_dropdown,
    )

    task = ctx.params.get("task")
    if task == "detect":
        coord_dropdown = types.Dropdown()
        coord_dropdown.add_choice(
            "normalized_1000", label="0-1000 (Qwen default)"
        )
        coord_dropdown.add_choice(
            "normalized_1", label="0-1 (relative)"
        )
        coord_dropdown.add_choice(
            "pixel", label="Pixel coordinates"
        )
        inputs.enum(
            "coordinate_format",
            coord_dropdown.values(),
            default="normalized_1000",
            label="Coordinate Format",
            view=coord_dropdown,
            description=(
                "Bounding box coordinate convention used by the model"
            ),
        )
