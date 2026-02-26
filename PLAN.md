# fo-vllm: FiftyOne + vLLM Plugin Integration Plan

## 1. Executive Summary

This plugin integrates vLLM with FiftyOne to provide universal VLM inference over image datasets. It supports **any model vLLM can serve** (Qwen2.5-VL, LLaVA, InternVL, Pixtral, Llama Vision, Phi-Vision, Gemma, etc.) in both **online** (remote/local server) and **offline** (in-process GPU) modes.

**Key principles**:

1. **Structured output only**: All non-text outputs use vLLM's constrained generation (`StructuredOutputsParams` with `choice` or `json`). Text parsing and fuzzy matching are never used.
2. **vLLM handles chat templates**: The plugin always sends OpenAI-format messages. vLLM automatically applies the correct chat template per model.
3. **Operates on FiftyOne datasets directly**: The operator works on `ctx.target_view()` (dataset, view, or selected samples) and writes results via `set_values()`.
4. **Parallel everything**: ThreadPool for image encoding, async for online API calls, native batching for offline mode.

**Core value**: One plugin, any VLM, any task, any scale.

---

## 2. Design Evaluation

Three designs were evaluated. The chosen approach cherry-picks the best aspects of each.

### Design A: Thin Operator with Direct Loop

The operator directly manages iteration over samples and calls vLLM.

| Aspect | Assessment |
|--------|-----------|
| Simplicity | High -- minimal abstraction |
| Reusability | Low -- logic locked inside operator |
| Parallelism | Manual -- must implement threading |

**Verdict**: Too monolithic. Mixing inference engine logic with FiftyOne operator logic hurts testability and reuse.

### Design B: FiftyOne Model Interface (`fo.Model`)

Implement FiftyOne's `Model` class and use `view.apply_model()`.

| Aspect | Assessment |
|--------|-----------|
| FiftyOne integration | Deep -- works with apply_model, Model Zoo |
| Image handling | Wasteful -- numpy decode/re-encode overhead |
| Prompt support | Awkward -- Model interface has no prompt concept |
| Batch efficiency | Moderate -- predict_all helps but images arrive as numpy |

**Verdict**: The `fo.Model` interface was designed for traditional vision models (input: tensor, output: label). VLMs need prompts, structured output constraints, and work best with file paths or base64 -- not numpy arrays.

### Design C: Hybrid Engine + Smart Operator (CHOSEN)

Separate `VLLMEngine` for inference, `TaskConfig` for prompts/structured output/parsing, and an operator for FiftyOne integration.

| Aspect | Assessment |
|--------|-----------|
| Image handling | Optimal -- file:// paths or parallel base64, no numpy |
| Batch efficiency | High -- native vLLM batching, bulk set_values() writes |
| Structured output | Full vLLM constrained generation (StructuredOutputsParams) |
| Separation of concerns | Clean -- engine, task config, and operator each have one job |

### Cherry-Picked Elements

| From | Element | Reason |
|------|---------|--------|
| Design A | Direct control over batch loop | Avoids apply_model overhead for VLMs |
| Design B | Reusable class external to operator | Engine can be used via SDK |
| Design C | Parallel image prep + bulk writes | Maximum throughput |
| Design C | File path-based image handling | Zero unnecessary image decoding |
| vLLM | `StructuredOutputsParams(choice=..., json=...)` | Enforced structured output, zero text parsing |

---

## 3. Architecture

```
                        fo-vllm plugin
 +---------------------------------------------------------+
 |                                                          |
 |  Operator (FiftyOne UI + execution)                      |
 |  +---------+  +-----------+  +----------+  +---------+   |
 |  | resolve |  | execute() |  | progress |  | resolve |   |
 |  | _input  |->| batching  |->| & save   |->| _output |   |
 |  +---------+  +-----+-----+  +----------+  +---------+   |
 |                      |                                   |
 |               +------v-------+                           |
 |               |  TaskConfig  |                           |
 |               | prompt build |                           |
 |               | JSON schemas |                           |
 |               | output parse |                           |
 |               +------+-------+                           |
 |                      |                                   |
 |               +------v-------+                           |
 |               |  VLLMEngine  |                           |
 |               |              |                           |
 |               | +----------+ |                           |
 |               | |  Online  | | OpenAI-compat async HTTP  |
 |               | +----------+ |                           |
 |               | +----------+ |                           |
 |               | | Offline  | | vllm.LLM.chat() batch     |
 |               | +----------+ |                           |
 |               +--------------+                           |
 +---------------------------------------------------------+
```

---

## 4. File Structure

```
fo-vllm/
  fiftyone.yml          # Plugin manifest
  __init__.py           # register(plugin), operator imports
  engine.py             # VLLMEngine: online/offline inference + structured output
  tasks.py              # TaskConfig: prompts, JSON schemas, output parsers
  operators.py          # FiftyOne operator: UI, batching, progress, result storage
  utils.py              # Image encoding, async helpers
  requirements.txt      # openai, vllm (optional), pillow
```

---

## 5. Task Taxonomy

The plugin supports 7 task types, selectable via a dropdown in the FiftyOne App. Each task defines three things: a **prompt template**, a **structured output constraint**, and an **output parser**.

### 5.1 Task Overview

| Task | Description | vLLM Constraint | FiftyOne Output Type |
|------|-------------|-----------------|---------------------|
| **Caption** | Generate image description | None (free text) | `StringField` |
| **Classify** | Single-label classification | `structured_outputs(choice=)` | `fo.Classification` |
| **Tag** | Multi-label tagging | `structured_outputs(json=)` | `fo.Classifications` |
| **Detect** | Object detection with bboxes | `structured_outputs(json=)` | `fo.Detections` |
| **VQA** | Visual question answering | None (free text) | `StringField` |
| **OCR** | Extract text from image | None (free text) | `StringField` |
| **Custom** | User-defined prompt | None (free text) | `StringField` |

### 5.2 Per-Task Specifications

#### Caption

- **Default prompt**: `"Describe this image concisely."`
- **System prompt**: None
- **Structured output**: None
- **Output parse**: Store raw text as `StringField`
- **Additional inputs**: None (prompt override available)

#### Classify

- **Default prompt**: `"Classify this image. Choose exactly one: {classes}"`
- **System prompt**: `"You are an image classifier. Respond with exactly one class label."`
- **Structured output**: `StructuredOutputsParams(choice=classes)` — forces output to be exactly one of the class names
- **Output parse**: `fo.Classification(label=output_text.strip())`
- **Additional inputs**: `classes` (required, comma-separated list)

#### Tag

- **Default prompt**: `"Tag this image with all applicable labels from: {classes}"`
- **System prompt**: `"You are an image tagger. Return a JSON object with a 'labels' array containing all applicable tags."`
- **Structured output**: `StructuredOutputsParams(json=schema)` with schema:

```json
{
  "type": "object",
  "properties": {
    "labels": {
      "type": "array",
      "items": {"type": "string", "enum": ["<class1>", "<class2>", "..."]}
    }
  },
  "required": ["labels"]
}
```

- **Output parse**: `json.loads(text)["labels"]` → `fo.Classifications(classifications=[fo.Classification(label=l) for l in labels])`
- **Additional inputs**: `classes` (required, comma-separated list)

#### Detect

- **Default prompt**: `"Detect all objects in this image. For each object, return its label and bounding box as [x_min, y_min, x_max, y_max] in 0-1000 coordinates."` (If classes provided: `"Detect these objects: {classes}..."`)
- **System prompt**: `"You are an object detector. Return a JSON object with a 'detections' array. Each detection has a 'label' string and 'box' array of [x_min, y_min, x_max, y_max] in 0-1000 coordinate space."`
- **Structured output**: `StructuredOutputsParams(json=schema)` with schema:

```json
{
  "type": "object",
  "properties": {
    "detections": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "label": {"type": "string"},
          "box": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4
          }
        },
        "required": ["label", "box"]
      }
    }
  },
  "required": ["detections"]
}
```

(When classes are provided, the `label` field adds `"enum": ["<class1>", ...]`)

- **Output parse**: `json.loads(text)["detections"]` → convert each `[x1, y1, x2, y2]` from 0-1000 to FiftyOne's `[x, y, w, h]` relative format (divide by 1000, convert xyxy→xywh) → `fo.Detections(detections=[fo.Detection(label=d["label"], bounding_box=[x, y, w, h]) for d in dets])`
- **Additional inputs**: `classes` (optional, comma-separated)
- **Caveat**: Coordinate accuracy depends on model capability. Works best with Qwen2.5-VL which natively supports grounding.

#### VQA

- **Default prompt**: `"{question}"`
- **System prompt**: None
- **Structured output**: None (free text)
- **Output parse**: Store raw text as `StringField`
- **Additional inputs**: `question` (required)

#### OCR

- **Default prompt**: `"Extract all text visible in this image. Return only the extracted text, nothing else."`
- **System prompt**: None
- **Structured output**: None
- **Output parse**: Store raw text as `StringField`
- **Additional inputs**: None (prompt override available)

#### Custom

- **Default prompt**: User-provided (required)
- **System prompt**: User-provided (optional)
- **Structured output**: None
- **Output parse**: Store raw text as `StringField`
- **Additional inputs**: `prompt` (required), `system_prompt` (optional)

### 5.3 Structured Output Strategy

**Core principle**: The plugin never parses free-form text to extract labels. Instead:

| Output type | Mechanism | Guarantee |
|-------------|-----------|-----------|
| Single label from fixed set | `structured_outputs(choice=)` | Output is exactly one of the allowed strings |
| JSON object/array | `structured_outputs(json=)` | Output is valid JSON conforming to the schema |
| Free text (captions, VQA, OCR) | No constraint | Stored as-is, no parsing needed |

The only "parsing" performed is `json.loads()` on JSON that has been schema-enforced by vLLM's constrained generation. This is deterministic and cannot fail (barring truncation from insufficient `max_tokens`).

**Engine mapping** (vLLM 0.16+ `StructuredOutputsParams` API):

- **Online mode**: `extra_body={"structured_outputs": {"choice": [...]}}` or `extra_body={"structured_outputs": {"json": schema}}`
- **Offline mode**: `StructuredOutputsParams(choice=[...])` or `StructuredOutputsParams(json=schema)` passed to `SamplingParams(structured_outputs=...)`

---

## 6. Component Specifications

### 6.1 `VLLMEngine` (`engine.py`)

Unified inference interface over both vLLM modes. Handles structured output constraints.

```python
class VLLMEngine:
    """Thin wrapper unifying online (API) and offline (in-process) vLLM."""

    def __init__(
        self,
        model: str,                             # HuggingFace model ID or path
        mode: str = "online",                   # "online" | "offline"
        # Online-only
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 64,               # Semaphore limit for async calls
        # Offline-only engine kwargs
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: dict | None = None,
        # Sampling parameters (both modes)
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        seed: int | None = None,
    ): ...

    def infer_batch(
        self,
        messages: list[list[dict]],
        structured_outputs: dict | None = None,
    ) -> list[str]:
        """Run batch inference with optional structured output constraints.

        Args:
            messages: list of OpenAI-format message lists, one per sample.
            structured_outputs: optional dict passed to vLLM's
                StructuredOutputsParams. Examples:
                  {"choice": ["cat", "dog"]}
                  {"json": {<JSON schema>}}
                Pass None for free-text tasks.

        Returns list of response strings (plain text or valid JSON).
        """
        if self.mode == "online":
            return self._online_batch(messages, structured_outputs)
        return self._offline_batch(messages, structured_outputs)

    def _online_batch(self, messages, structured_outputs) -> list[str]:
        """Async concurrent OpenAI API calls with semaphore for backpressure."""
        # Build extra_body from structured output params
        # extra_body = {}
        # if structured_outputs:
        #     extra_body["structured_outputs"] = structured_outputs
        #
        # sem = asyncio.Semaphore(self._max_concurrent)
        # async def _call(msgs):
        #     async with sem:
        #         resp = await self._client.chat.completions.create(
        #             model=self.model, messages=msgs,
        #             temperature=self.temperature, max_tokens=self.max_tokens,
        #             top_p=self.top_p, seed=self.seed,
        #             extra_body=extra_body or None,
        #         )
        #         return resp.choices[0].message.content
        # return asyncio.run(asyncio.gather(*[_call(m) for m in messages]))
        ...

    def _offline_batch(self, messages, structured_outputs) -> list[str]:
        """vLLM LLM.chat() native batch inference with structured output."""
        # from vllm import SamplingParams
        # from vllm.sampling_params import StructuredOutputsParams
        #
        # so_params = None
        # if structured_outputs:
        #     so_params = StructuredOutputsParams(**structured_outputs)
        #
        # params = SamplingParams(
        #     temperature=self.temperature, max_tokens=self.max_tokens,
        #     top_p=self.top_p, seed=self.seed,
        #     structured_outputs=so_params,
        # )
        # outputs = self._llm.chat(messages, sampling_params=params)
        # return [o.outputs[0].text for o in outputs]
        ...

    def cleanup(self):
        """Free GPU memory (offline mode)."""
        if hasattr(self, "_llm") and self._llm is not None:
            del self._llm
            self._llm = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

### 6.2 `TaskConfig` (`tasks.py`)

Handles prompt construction, structured output schema generation, and deterministic output parsing per task type.

```python
import json
import fiftyone as fo


class TaskConfig:
    """Builds prompts, structured output constraints, and parses VLM responses."""

    TASKS = {
        "caption": {
            "system": None,
            "prompt": "Describe this image concisely.",
            "output_type": "string",     # FiftyOne field type
        },
        "classify": {
            "system": "You are an image classifier. Respond with exactly one class label.",
            "prompt": "Classify this image. Choose exactly one: {classes}",
            "output_type": "Classification",
        },
        "tag": {
            "system": (
                "You are an image tagger. Return a JSON object with a "
                "'labels' array containing all applicable tags."
            ),
            "prompt": "Tag this image with all applicable labels from: {classes}",
            "output_type": "Classifications",
        },
        "detect": {
            "system": (
                "You are an object detector. Return a JSON object with a "
                "'detections' array. Each detection has a 'label' string "
                "and 'box' array of [x_min, y_min, x_max, y_max] in "
                "0-1000 coordinate space."
            ),
            "prompt": "Detect all objects in this image.",
            "output_type": "Detections",
        },
        "vqa": {
            "system": None,
            "prompt": "{question}",
            "output_type": "string",
        },
        "ocr": {
            "system": None,
            "prompt": (
                "Extract all text visible in this image. "
                "Return only the extracted text, nothing else."
            ),
            "output_type": "string",
        },
        "custom": {
            "system": None,
            "prompt": "{prompt}",
            "output_type": "string",
        },
    }

    def __init__(
        self,
        task: str,
        prompt: str | None = None,
        system_prompt: str | None = None,
        classes: list[str] | None = None,
        **template_kwargs,
    ):
        defaults = self.TASKS[task]
        self.task = task
        self.classes = classes
        self.output_type = defaults["output_type"]
        self.system_prompt = system_prompt if system_prompt is not None else defaults["system"]

        # Build the user prompt from template
        raw_prompt = prompt if prompt is not None else defaults["prompt"]
        fmt_kwargs = {**template_kwargs}
        if classes:
            fmt_kwargs["classes"] = ", ".join(classes)
        self.prompt = raw_prompt.format(**fmt_kwargs) if fmt_kwargs else raw_prompt

    def build_messages(self, image_content: dict) -> list[dict]:
        """Build OpenAI-format messages for one image."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": [image_content, {"type": "text", "text": self.prompt}],
        })
        return messages

    # -- Structured output constraints (vLLM 0.16+ StructuredOutputsParams) --

    def get_structured_outputs(self) -> dict | None:
        """Return kwargs dict for StructuredOutputsParams, or None.

        Used as:
          - Online: extra_body={"structured_outputs": result}
          - Offline: StructuredOutputsParams(**result) → SamplingParams(structured_outputs=...)
        """
        if self.task == "classify" and self.classes:
            return {"choice": self.classes}

        if self.task == "tag" and self.classes:
            return {
                "json": {
                    "type": "object",
                    "properties": {
                        "labels": {
                            "type": "array",
                            "items": {"type": "string", "enum": self.classes},
                        }
                    },
                    "required": ["labels"],
                }
            }

        if self.task == "detect":
            label_schema = {"type": "string"}
            if self.classes:
                label_schema["enum"] = self.classes
            return {
                "json": {
                    "type": "object",
                    "properties": {
                        "detections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": label_schema,
                                    "box": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                },
                                "required": ["label", "box"],
                            },
                        }
                    },
                    "required": ["detections"],
                }
            }

        return None

    # -- Output parsing --

    def parse_response(self, text: str) -> fo.Classification | fo.Classifications | fo.Detections | str:
        """Parse VLM response into a FiftyOne label or string.

        For structured outputs (classify, tag, detect), the text is
        guaranteed to be well-formed by vLLM's constrained generation.
        """
        if self.output_type == "string":
            return text.strip()

        if self.output_type == "Classification":
            return fo.Classification(label=text.strip())

        if self.output_type == "Classifications":
            data = json.loads(text)
            return fo.Classifications(
                classifications=[
                    fo.Classification(label=label)
                    for label in data["labels"]
                ]
            )

        if self.output_type == "Detections":
            data = json.loads(text)
            detections = []
            for det in data["detections"]:
                x1, y1, x2, y2 = det["box"]
                # Convert from 0-1000 xyxy to FiftyOne's [0,1] xywh
                x = x1 / 1000.0
                y = y1 / 1000.0
                w = (x2 - x1) / 1000.0
                h = (y2 - y1) / 1000.0
                detections.append(
                    fo.Detection(
                        label=det["label"],
                        bounding_box=[x, y, w, h],
                    )
                )
            return fo.Detections(detections=detections)

        return text.strip()
```

### 6.3 Image Content Builder (`utils.py`)

Parallel construction of image content objects from file paths.

```python
def build_image_contents(
    filepaths: list[str],
    image_mode: str = "auto",        # "auto" | "base64" | "filepath"
    base_url: str | None = None,     # Used by "auto" to detect local server
    max_workers: int | None = None,
) -> list[dict]:
    """Convert file paths to OpenAI image_url content dicts. Parallelized.

    Returns list of: {"type":"image_url","image_url":{"url":"..."}}
    """
    ...
```

- `"filepath"` mode: No I/O. Builds `{"url": "file:///absolute/path.jpg"}`. Instant.
- `"base64"` mode: `ThreadPoolExecutor` reads + base64-encodes files in parallel.
- `"auto"` mode: If `base_url` contains `localhost` or `127.0.0.1`, use filepath. Otherwise, base64.

### 6.4 Operator (`operators.py`)

Single operator for all VLM inference tasks. The `resolve_input` method uses FiftyOne's `types.*` API to build a dynamic form.

```python
import fiftyone.operators as foo
from fiftyone.operators import types


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

        mode = _mode_selector(ctx, inputs)
        _model_selector(ctx, inputs)
        if mode == "online":
            _server_settings(ctx, inputs)
        task = _task_selector(ctx, inputs)
        _task_settings(ctx, inputs, task)
        _output_settings(ctx, inputs, task)
        _advanced_settings(ctx, inputs, mode)
        inputs.view_target(ctx)

        return types.Property(inputs, view=types.View(label="vLLM Inference"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        # 1. Build engine from params
        engine = VLLMEngine(
            model=ctx.params["model"],
            mode=ctx.params["mode"],
            base_url=ctx.params.get("base_url", "http://localhost:8000/v1"),
            api_key=ctx.params.get("api_key", "EMPTY"),
            temperature=ctx.params.get("temperature", 0.0),
            max_tokens=ctx.params.get("max_tokens", 512),
            top_p=ctx.params.get("top_p", 1.0),
            # Offline-only
            tensor_parallel_size=ctx.params.get("tensor_parallel_size", 1),
            gpu_memory_utilization=ctx.params.get("gpu_memory_utilization", 0.9),
            max_model_len=ctx.params.get("max_model_len") or None,
        )

        try:
            # 2. Build task config
            classes = None
            raw_classes = ctx.params.get("classes")
            if raw_classes:
                classes = [c.strip() for c in raw_classes.split(",")]

            task = TaskConfig(
                task=ctx.params["task"],
                prompt=ctx.params.get("prompt_override") or ctx.params.get("prompt"),
                system_prompt=ctx.params.get("system_prompt"),
                classes=classes,
                question=ctx.params.get("question", ""),
            )

            # 3. Get target samples
            view = ctx.target_view()
            ids = view.values("id")
            filepaths = view.values("filepath")
            total = len(ids)
            batch_size = ctx.params.get("batch_size", 32)
            output_field = ctx.params["output_field"]

            # Get structured output constraints from task
            structured_outputs = task.get_structured_outputs()

            # 4. Process in batches
            processed = 0
            for i in range(0, total, batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_paths = filepaths[i:i + batch_size]

                # 4a. Parallel image content construction
                image_contents = build_image_contents(
                    batch_paths,
                    image_mode=ctx.params.get("image_mode", "auto"),
                    base_url=ctx.params.get("base_url"),
                )

                # 4b. Build messages for each image
                batch_messages = [task.build_messages(img) for img in image_contents]

                # 4c. Batch inference with structured output
                responses = engine.infer_batch(
                    batch_messages,
                    structured_outputs=structured_outputs,
                )

                # 4d. Parse responses and bulk-write to dataset
                results = {
                    sid: task.parse_response(resp)
                    for sid, resp in zip(batch_ids, responses)
                }
                ctx.dataset.set_values(output_field, results, key_field="id")

                # 4e. Progress
                processed += len(batch_ids)
                if ctx.delegated:
                    ctx.set_progress(
                        progress=processed / total,
                        label=f"{processed}/{total} samples",
                    )
                else:
                    yield ctx.trigger(
                        "set_progress",
                        dict(progress=processed / total,
                             label=f"{processed}/{total} samples"),
                    )

            # 5. Reload dataset in App
            if not ctx.delegated:
                yield ctx.trigger("reload_dataset")

        finally:
            engine.cleanup()

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("summary", label="Summary")
        return types.Property(outputs, view=types.View(label="Complete"))
```

---

## 7. Operator UI Specification

All helper functions used in `resolve_input()` are defined below with their exact FiftyOne `types.*` API usage.

### 7.1 `_mode_selector(ctx, inputs)`

Inference mode: Online (API server) or Offline (local vLLM).

```python
def _mode_selector(ctx, inputs):
    mode_radio = types.RadioGroup(orientation="horizontal")
    mode_radio.add_choice(
        "online",
        label="Online",
        description="Connect to a vLLM API server",
    )
    mode_radio.add_choice(
        "offline",
        label="Offline",
        description="Run vLLM locally (requires GPU + vllm package)",
    )
    inputs.enum(
        "mode",
        mode_radio.values(),
        default="online",
        required=True,
        label="Inference Mode",
        view=mode_radio,
    )
    return ctx.params.get("mode", "online")
```

**Produces**: `ctx.params["mode"]` → `"online"` or `"offline"`

### 7.2 `_model_selector(ctx, inputs)`

Model ID/name input.

```python
def _model_selector(ctx, inputs):
    inputs.str(
        "model",
        label="Model",
        required=True,
        description="HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)",
    )
```

**Produces**: `ctx.params["model"]` → `"Qwen/Qwen2.5-VL-7B-Instruct"`

### 7.3 `_server_settings(ctx, inputs)`

Server connection settings (shown only when mode is "online").

```python
def _server_settings(ctx, inputs):
    inputs.view("server_header", types.Header(label="Server Settings", divider=True))
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
```

**Produces**: `ctx.params["base_url"]`, `ctx.params["api_key"]`

**Conditional display**: Only rendered when `mode == "online"` (controlled by the `if mode == "online":` check in `resolve_input`).

### 7.4 `_task_selector(ctx, inputs)`

Task type dropdown (single-select).

```python
def _task_selector(ctx, inputs):
    task_dropdown = types.Dropdown(label="Task")
    task_dropdown.add_choice(
        "caption", label="Caption",
        description="Generate a text description of the image",
    )
    task_dropdown.add_choice(
        "classify", label="Classify",
        description="Assign a single class label (constrained output)",
    )
    task_dropdown.add_choice(
        "tag", label="Tag",
        description="Assign multiple labels (constrained JSON output)",
    )
    task_dropdown.add_choice(
        "detect", label="Detect",
        description="Detect objects with bounding boxes (constrained JSON output)",
    )
    task_dropdown.add_choice(
        "vqa", label="VQA",
        description="Answer a question about the image",
    )
    task_dropdown.add_choice(
        "ocr", label="OCR",
        description="Extract text visible in the image",
    )
    task_dropdown.add_choice(
        "custom", label="Custom",
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
```

**Produces**: `ctx.params["task"]` → one of `"caption"`, `"classify"`, `"tag"`, `"detect"`, `"vqa"`, `"ocr"`, `"custom"`

### 7.5 `_task_settings(ctx, inputs, task)`

Conditional fields that appear based on the selected task.

```python
def _task_settings(ctx, inputs, task):
    if task is None:
        return

    inputs.view("task_header", types.Header(label="Task Settings", divider=True))

    # Classes input (classify, tag, detect)
    if task in ("classify", "tag"):
        inputs.str(
            "classes",
            label="Classes",
            required=True,
            description="Comma-separated class names (e.g., cat, dog, bird)",
        )
    elif task == "detect":
        inputs.str(
            "classes",
            label="Classes",
            required=False,
            description="Optional: comma-separated class names to detect (leave empty for open detection)",
        )

    # Question input (VQA)
    if task == "vqa":
        inputs.str(
            "question",
            label="Question",
            required=True,
            description="Question to ask about each image",
        )

    # Custom prompt input
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

    # Prompt override for non-custom tasks
    if task not in ("custom",):
        inputs.str(
            "prompt_override",
            label="Prompt Override",
            required=False,
            description="Override the default prompt for this task",
        )
```

**Conditional display logic**:

| Task | Fields shown |
|------|-------------|
| `caption` | prompt_override |
| `classify` | classes (required), prompt_override |
| `tag` | classes (required), prompt_override |
| `detect` | classes (optional), prompt_override |
| `vqa` | question (required), prompt_override |
| `ocr` | prompt_override |
| `custom` | prompt (required), system_prompt |

### 7.6 `_output_settings(ctx, inputs, task)`

Output field name with task-appropriate defaults.

```python
def _output_settings(ctx, inputs, task):
    defaults = {
        "caption": "caption",
        "classify": "classification",
        "tag": "tags",
        "detect": "detections",
        "vqa": "vqa_answer",
        "ocr": "ocr_text",
        "custom": "vlm_output",
    }
    inputs.str(
        "output_field",
        label="Output Field",
        required=True,
        default=defaults.get(task, "vlm_output"),
        description="Field name to store results on each sample",
    )
```

**Produces**: `ctx.params["output_field"]` → field name string

### 7.7 `_advanced_settings(ctx, inputs, mode)`

Advanced settings behind a toggle.

```python
def _advanced_settings(ctx, inputs, mode):
    inputs.view("adv_header", types.Header(label="Advanced Settings", divider=True))
    inputs.bool(
        "show_advanced",
        label="Show advanced settings",
        default=False,
        view=types.SwitchView(),
    )

    if not ctx.params.get("show_advanced", False):
        return

    # -- Sampling parameters (all modes) --
    inputs.float(
        "temperature", label="Temperature",
        default=0.0, min=0.0, max=2.0,
        description="Sampling temperature (0.0 = deterministic)",
    )
    inputs.int(
        "max_tokens", label="Max Tokens",
        default=512, min=1, max=4096,
        description="Maximum tokens to generate per sample",
    )
    inputs.float(
        "top_p", label="Top P",
        default=1.0, min=0.0, max=1.0,
    )
    inputs.int(
        "batch_size", label="Batch Size",
        default=32, min=1, max=512,
        description="Number of samples per inference batch",
    )

    # -- Image handling --
    image_dropdown = types.Dropdown()
    image_dropdown.add_choice("auto", label="Auto", description="Detect based on server URL")
    image_dropdown.add_choice("filepath", label="File Path", description="file:// paths (local servers only)")
    image_dropdown.add_choice("base64", label="Base64", description="Base64-encoded (works everywhere)")
    inputs.enum(
        "image_mode",
        image_dropdown.values(),
        default="auto",
        label="Image Mode",
        view=image_dropdown,
    )

    # -- Offline-only engine settings --
    if mode == "offline":
        inputs.view("offline_header", types.Header(label="Offline Engine Settings", divider=True))
        inputs.int(
            "tensor_parallel_size", label="Tensor Parallel Size",
            default=1, min=1, max=8,
            description="Number of GPUs for tensor parallelism",
        )
        inputs.float(
            "gpu_memory_utilization", label="GPU Memory Utilization",
            default=0.9, min=0.1, max=1.0,
            description="Fraction of GPU memory to use",
        )
        inputs.int(
            "max_model_len", label="Max Model Length",
            default=0, min=0, max=131072,
            description="Max context length (0 = model default)",
        )
```

**Produces** (when expanded): `ctx.params["temperature"]`, `ctx.params["max_tokens"]`, `ctx.params["top_p"]`, `ctx.params["batch_size"]`, `ctx.params["image_mode"]`, plus offline-specific params.

---

## 8. Critical Technical Decisions

### 8.1 Chat Templates Are Handled by vLLM

Both `LLM.chat()` (offline) and `/v1/chat/completions` (online) automatically apply the correct Jinja2 chat template per model. The plugin always uses the universal OpenAI message format:

```python
[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "..."}},
    {"type": "text", "text": "..."},
]}]
```

### 8.2 Structured Output Eliminates Text Parsing

| Task | Constraint | Why no parsing needed |
|------|-----------|----------------------|
| Classify | `StructuredOutputsParams(choice=classes)` | Output is exactly one class name |
| Tag | `StructuredOutputsParams(json=schema)` | Output is valid JSON with enum-constrained labels |
| Detect | `StructuredOutputsParams(json=schema)` | Output is valid JSON with label+box arrays |
| Caption/VQA/OCR/Custom | None | Output is free text, stored as-is |

### 8.3 Image Transfer Strategy

| Scenario | Strategy | Why |
|----------|----------|-----|
| Offline mode | `file://` paths | vLLM loads directly, zero copy |
| Online, local server | `file://` paths (with `--allowed-local-media-path`) | Zero copy |
| Online, remote server | Parallel base64 encoding via ThreadPool | Only option |

Auto-detection: if `base_url` contains `localhost` or `127.0.0.1`, use file paths. Otherwise, base64.

### 8.4 Bulk Writes via `set_values()`

`dataset.set_values(field, {id: value, ...}, key_field="id")` performs a single bulk MongoDB write per batch. Dramatically faster than per-sample `save()`.

### 8.5 Offline Mode: Single LLM Instance

The `vllm.LLM` object is instantiated once and reused for all batches. Cleaned up in `engine.cleanup()` via the operator's `finally` block.

### 8.6 Online Mode: Async Concurrency

`AsyncOpenAI` with `asyncio.Semaphore(max_concurrent)`. All requests in a batch are fired concurrently. The semaphore prevents overwhelming the server. vLLM's continuous batching handles the rest.

### 8.7 Detection Coordinate Convention

The plugin prompts for and expects 0-1000 coordinate space (matching Qwen2.5-VL's native grounding format). Conversion to FiftyOne's `[x, y, w, h]` relative format (0-1 range): divide by 1000, convert xyxy→xywh.

---

## 9. SDK API

Beyond the operator, users can invoke from Python:

```python
import fiftyone as fo
import fiftyone.operators as foo

# Via operator execution
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "mode": "online",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "classify",
        "classes": "cat, dog, bird",
        "output_field": "animal_class",
        "batch_size": 64,
    },
    dataset_name="my_dataset",
)

# Or directly via the engine (advanced usage)
from fo_vllm.engine import VLLMEngine
from fo_vllm.tasks import TaskConfig
from fo_vllm.utils import build_image_contents

dataset = fo.load_dataset("my_dataset")
engine = VLLMEngine(model="Qwen/Qwen2.5-VL-7B-Instruct", mode="online")
task = TaskConfig(task="classify", classes=["indoor", "outdoor"])

filepaths = dataset.values("filepath")
images = build_image_contents(filepaths, image_mode="base64")
messages = [task.build_messages(img) for img in images]
responses = engine.infer_batch(
    messages,
    structured_outputs=task.get_structured_outputs(),
)

results = {sid: task.parse_response(r) for sid, r in zip(dataset.values("id"), responses)}
dataset.set_values("scene_type", results, key_field="id")
engine.cleanup()
```

---

## 10. Implementation Plan

### Phase 1: Core Components (Parallelizable: 2 tracks)

**Track A: `engine.py` + `utils.py`**

1. `VLLMEngine.__init__()` for online mode (AsyncOpenAI client setup)
2. `_online_batch()` with async concurrency, semaphore, `structured_outputs` via `extra_body`
3. `build_image_contents()` — filepath mode (string formatting, no I/O)
4. `build_image_contents()` — base64 mode (ThreadPoolExecutor parallel I/O)
5. `build_image_contents()` — auto-detection logic
6. `VLLMEngine.__init__()` for offline mode (vLLM LLM instantiation, lazy import)
7. `_offline_batch()` using `LLM.chat()` with `StructuredOutputsParams`
8. `cleanup()` with GPU memory freeing

**Track B: `tasks.py`**

1. `TaskConfig.__init__()` with 7 task defaults and prompt template formatting
2. `build_messages()` — OpenAI message format construction
3. `get_structured_outputs()` — returns `StructuredOutputsParams` kwargs for classify (choice), tag (json), detect (json)
5. `parse_response()` for string output (caption, vqa, ocr, custom)
6. `parse_response()` for Classification (classify — direct label from structured choice)
7. `parse_response()` for Classifications (tag — JSON parse to fo.Classifications)
8. `parse_response()` for Detections (detect — JSON parse + coordinate conversion to fo.Detections)

### Phase 2: FiftyOne Integration

**Track C: `operators.py` + `__init__.py` + `fiftyone.yml`** (depends on Phase 1)

1. `fiftyone.yml` manifest
2. `_mode_selector()` — RadioGroup for online/offline
3. `_model_selector()` — text input for model ID
4. `_server_settings()` — base_url and api_key fields
5. `_task_selector()` — Dropdown with 7 task choices
6. `_task_settings()` — conditional fields per task (classes, question, prompt, system_prompt, prompt_override)
7. `_output_settings()` — output field name with task-specific defaults
8. `_advanced_settings()` — SwitchView toggle, sampling params, image mode, offline engine settings
9. `execute()` — batch loop, image prep, inference, parsing, set_values, progress
10. `resolve_delegation()` and `resolve_output()`
11. `__init__.py` with `register()` function
12. `requirements.txt`

### Phase 3: Testing

1. Online mode: classify task with structured_outputs(choice=) against a vLLM server
2. Online mode: tag task with structured_outputs(json=)
3. Online mode: caption task (free text)
4. Online mode: detect task with structured_outputs(json=)
5. Offline mode: classify + tag + caption with local vLLM
6. Progress reporting (immediate and delegated)
7. Base64 vs filepath image modes
8. Edge cases: empty dataset, missing files, model errors, max_tokens too low for JSON

### Parallelization Summary

```
Phase 1:  Track A (engine + utils) ──────────┐
          Track B (tasks)           ──────────┤
                                              v
Phase 2:  Track C (operator + plugin wiring) ─┐
                                               v
Phase 3:  Testing ─────────────────────────────>
```

Tracks A and B have zero dependencies and can be built simultaneously.

---

## 11. Plugin Manifest + Requirements

### `fiftyone.yml`

```yaml
name: "@fo-vllm/vllm"
type: plugin
author: "fo-vllm contributors"
version: "0.1.0"
description: "Universal VLM inference via vLLM -- any model, any task, online or offline"
fiftyone:
  version: ">=0.25"
operators:
  - run_vllm_inference
secrets:
  - FIFTYONE_VLLM_API_KEY
  - FIFTYONE_VLLM_BASE_URL
```

### `requirements.txt`

```
openai>=1.0
vllm>=0.8.5
pillow>=9.0
```

Note: `vllm>=0.8.5` is required for the `StructuredOutputsParams` API used by this plugin. `vllm` is only required for offline mode. Online mode only needs `openai`. The plugin should handle the case where `vllm` is not installed and restrict to online-only mode with a clear message.

---

## 12. Future Considerations (Post-MVP)

| Feature | How Architecture Supports It |
|---------|------------------------------|
| **Structured JSON extraction** | Add a `"json_extract"` task to TaskConfig with user-provided JSON schema passed to `structured_outputs(json=)` |
| **Multi-image per sample** | Extend `build_messages()` to accept multiple images per sample |
| **Video understanding** | Add video frame extraction to utils; some VLMs support video inputs |
| **Per-sample prompts from field** | Read prompt per-sample from a dataset field (e.g., `sample["question"]`) |
| **LoRA adapter selection** | vLLM supports runtime LoRA loading; add adapter parameter to engine |
| **Model comparison** | Run multiple models and store results in different fields |
| **Embeddings extraction** | Add embedding extraction mode for VLMs that expose embeddings |
| **FiftyOne Model Zoo** | Wrap VLLMEngine in `fo.Model` for `foz.load_zoo_model()` compat |
| **Panel UI** | React panel for interactive VLM chat with selected images |
| **Streaming responses** | vLLM streaming for interactive panel use |
| **Keypoint estimation** | Add `"keypoint"` task with structured_outputs(json=) schema → `fo.Keypoints` |
| **Segmentation polygons** | Add `"segment"` task with structured_outputs(json=) schema → `fo.Polylines` |
| **Regression / scoring** | Add `"score"` task with structured_outputs(json=) numeric schema → `fo.Regression` |
| **Content moderation** | Classify task with predefined NSFW/safety classes |
| **Label verification** | Classify task comparing VLM output to existing labels |
| **Hierarchical classification** | structured_outputs(json=) with multi-level schema |
| **Model auto-discovery** | Query `/v1/models` endpoint to populate model dropdown dynamically |

### Architectural hooks

1. **TaskConfig is extensible**: New tasks add an entry to `TASKS`, a `get_structured_outputs()` branch, and a `parse_response()` branch. Engine and operator are untouched.
2. **VLLMEngine is mode-agnostic**: New vLLM features (LoRA, speculative decoding, quantization) only touch `engine.py`.
3. **Operator UI is dynamic**: New parameters only require changes to the relevant helper function in `operators.py`.

---

## 13. Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| vLLM not installed (offline mode) | Lazy import with clear error message; online mode still works |
| Server unreachable (online mode) | Connection test before batch processing; clear error |
| Model OOM (offline mode) | Expose `gpu_memory_utilization` and `max_model_len` in advanced settings |
| Slow base64 encoding | ThreadPool parallelism; optional max image dimension resize |
| `max_tokens` too low for JSON output | Document minimum recommended values per task; validate in resolve_input |
| `structured_outputs` not supported by model/backend | Fallback to prompting-only mode with `json.loads()` attempt + raw text fallback |
| Detection coordinate mismatch | Document 0-1000 convention; clamp values to [0, 1000] in parser |
| Dataset too large for memory | Stream IDs/filepaths with batch iteration; never load all samples |
