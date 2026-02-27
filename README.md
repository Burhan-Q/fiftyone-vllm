# fo-vllm

FiftyOne plugin for VLM inference via vLLM. One operator, any model, any task.

Works with any vision-language model that vLLM can serve: Qwen2.5-VL, LLaVA, InternVL, Pixtral, Llama Vision, Phi-Vision, Gemma, and others.

## Installation

```shell
fiftyone plugins download https://github.com/<org>/fo-vllm
```

Or install locally:

```shell
fiftyone plugins create /path/to/fo-vllm
```

### Quick-start with Docker

A `compose.yml` is included for launching a vLLM server:

```shell
# Set your HuggingFace token
export HF_TOK=hf_...

# Start vLLM with Qwen2.5-VL
docker compose up -d
```

The default configuration serves `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` on port 8811. Edit `compose.yml` to change the model, GPU memory utilization, or context length.

## Architecture

The plugin follows a **VLLMEngine + TaskConfig + Operator** design:

- **`VLLMEngine`** (`engine.py`) — Async HTTP client wrapping vLLM's OpenAI-compatible API. Uses `AsyncOpenAI` with `asyncio.Semaphore` for concurrency control.
- **`TaskConfig`** (`tasks.py`) — Prompt templates, JSON schema constraints, and output parsers for each task. Every task uses vLLM structured output (`StructuredOutputsParams`) to guarantee valid responses at the token level.
- **`VLLMInference`** (`operators.py`) — Single FiftyOne operator that wires everything together: UI, batching, progress tracking, and result storage.
- **`utils.py`** — Image loading/encoding utilities (base64 or file:// paths).

## Tasks

| Task | Output | FiftyOne Type | Structured Output |
|------|--------|---------------|-------------------|
| **Caption** | text description | `fo.Classification(label=text)` | `json: {"text": string}` |
| **Classify** | single label | `fo.Classification(label=class)` | `choice: [classes]` |
| **Tag** | multiple labels | `fo.Classifications` | `json: {"labels": [enum]}` |
| **Detect** | bounding boxes | `fo.Detections` | `json: {"detections": [{label, box}]}` |
| **VQA** | answer text | `fo.Classification(label=answer)` | `json: {"answer": string}` |
| **OCR** | extracted text | `fo.Classification(label=text)` | `json: {"text": string}` |
| **Custom** | free-form response | `fo.Classification(label=response)` | `json: {"response": string}` |

All tasks return FiftyOne label types, which enables dynamic attribute attachment for metadata logging. There is no regex or free-text parsing anywhere in the plugin.

## Usage via the FiftyOne App

1. Open a dataset in the FiftyOne App
2. Press the `+` button in the operator browser or use the backtick (`` ` ``) shortcut
3. Search for **Run vLLM Inference**
4. Fill in the model ID, server URL, task, and task-specific options
5. Choose to run immediately or delegate to background execution

## Usage via Python SDK

All examples assume a vLLM server is running and accessible.

### Caption a dataset

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("my-images")

foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "caption",
    },
    dataset_name=dataset.name,
)

# Results stored as fo.Classification on an auto-named field
print(dataset.first().vllm_infer_caption.label)
```

### Classify images

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "classify",
        "classes": "indoor, outdoor, aerial",
    },
    dataset_name="my-images",
)

# fo.Classification label (constrained to one of the provided classes)
print(dataset.first().vllm_infer_classification.label)
```

### Multi-label tagging

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "tag",
        "classes": "person, vehicle, animal, building, nature, food",
    },
    dataset_name="my-images",
)

# fo.Classifications with multiple labels
for c in dataset.first().vllm_infer_tags.classifications:
    print(c.label)
```

### Object detection

```python
# Open detection (no class constraint)
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "detect",
    },
    dataset_name="my-images",
)

# Class-constrained detection
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "detect",
        "classes": "car, truck, bus, motorcycle, bicycle",
    },
    dataset_name="my-images",
)
```

### Visual question answering

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "vqa",
        "question": "How many people are in this image?",
    },
    dataset_name="my-images",
)

print(dataset.first().vllm_infer_vqa_answer.label)
```

### OCR

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "ocr",
    },
    dataset_name="my-images",
)

print(dataset.first().vllm_infer_ocr_text.label)
```

### Custom prompt

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "custom",
        "prompt": "Is this image safe for work? Explain briefly.",
        "system_prompt": "You are a content moderator.",
    },
    dataset_name="my-images",
)

print(dataset.first().vllm_infer_vlm_output.label)
```

### Override the default prompt

Every built-in task has a default prompt. Override it for any task except Custom:

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "caption",
        "prompt_override": "Describe this image in exactly one sentence, focusing on the dominant colors.",
    },
    dataset_name="my-images",
)
```

### Log run metadata

Enable metadata logging to record model name, prompt, and inference config on each result:

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "caption",
        "log_metadata": True,
    },
    dataset_name="my-images",
)

# Dynamic attributes on each label
label = dataset.first().vllm_infer_caption
print(label.model_name)   # "Qwen/Qwen2.5-VL-7B-Instruct"
print(label.prompt)        # "[system] ...\n[user] ..."
print(label.infer_cfg)     # {"temperature": 0.2, "max_tokens": 512, ...}

# Run metadata also stored at dataset level
print(dataset.info["vllm_runs"]["vllm_infer_caption"])
```

### Overwrite previous results

By default, running the same task creates a new field with an incremented suffix (`vllm_infer_caption`, `vllm_infer_caption1`, ...). To overwrite the most recent result instead:

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "caption",
        "overwrite_last": True,
    },
    dataset_name="my-images",
)
```

## Direct engine usage

For advanced workflows outside the operator, use the engine and task config directly:

```python
import fiftyone as fo

from fo_vllm.engine import VLLMEngine
from fo_vllm.tasks import TaskConfig
from fo_vllm.utils import build_image_contents

dataset = fo.load_dataset("my-images")

engine = VLLMEngine(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    base_url="http://localhost:8000/v1",
)
task = TaskConfig(task="classify", classes=["indoor", "outdoor"])

filepaths = dataset.values("filepath")
images = build_image_contents(filepaths, image_mode="auto")
messages = [task.build_messages(img) for img in images]
responses = engine.infer_batch(
    messages,
    structured_outputs=task.get_structured_outputs(),
)

results = {}
for sid, r in zip(dataset.values("id"), responses):
    if isinstance(r, Exception):
        continue
    results[sid] = task.parse_response(r)

dataset.set_values("scene_type", results, key_field="id")
```

## Output field naming

Results are stored as flat sample fields with auto-generated names based on the task's default field:

| Task | Default field name |
|------|-------------------|
| caption | `vllm_infer_caption` |
| classify | `vllm_infer_classification` |
| tag | `vllm_infer_tags` |
| detect | `vllm_infer_detections` |
| vqa | `vllm_infer_vqa_answer` |
| ocr | `vllm_infer_ocr_text` |
| custom | `vllm_infer_vlm_output` |

Subsequent runs of the same task auto-increment: `vllm_infer_caption`, `vllm_infer_caption1`, `vllm_infer_caption2`, etc. Enable "Overwrite last result" to re-use the most recent field instead.

## Configuration

### Server settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | (required) | HuggingFace model ID served by vLLM |
| `base_url` | `http://localhost:8000/v1` | vLLM OpenAI-compatible API endpoint |
| `api_key` | `EMPTY` | API key for authentication |

Server settings can also be configured via FiftyOne secrets:

- `FIFTYONE_VLLM_API_KEY` — API key
- `FIFTYONE_VLLM_BASE_URL` — Server URL

Precedence: UI parameter > FiftyOne secret > default.

Server settings are persisted in `dataset.info["_vllm_config"]` and restored as defaults on subsequent runs.

### Advanced settings

Toggle "Show advanced settings" in the operator form to access these:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | task-specific | Sampling temperature (0.0 for deterministic tasks, 0.2 for generative). Leave empty for task default. |
| `max_tokens` | 512 | Maximum tokens per response |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `batch_size` | 8 | Samples per inference batch |
| `max_concurrent` | 16 | Parallel HTTP requests to vLLM |
| `max_workers` | 4 | Threads for image loading/encoding |
| `image_mode` | auto | Image transfer method: `auto` or `filepath` |
| `coordinate_format` | normalized_1000 | Detection coordinate convention (detect task only) |

### Image mode

| Mode | When to use |
|------|-------------|
| `auto` | Default. URLs pass through directly; local files are base64-encoded in parallel. |
| `filepath` | Local vLLM server with `--allowed-local-media-path`. Sends `file://` URLs with zero I/O overhead. |

### Detection coordinate format

Different VLMs use different coordinate conventions for bounding boxes:

| Format | Range | Models |
|--------|-------|--------|
| `normalized_1000` | 0--1000 | Qwen2-VL, Qwen2.5-VL, Qwen3-VL (default) |
| `normalized_1` | 0.0--1.0 | Some fine-tuned models |
| `pixel` | 0--image_dim | InternVL, others |

The plugin converts all formats to FiftyOne's `[x, y, w, h]` relative coordinates automatically. Post-generation validation catches degenerate boxes (zero or negative dimensions) and clamps coordinates to valid ranges.

## Delegated execution

The operator supports FiftyOne's delegated execution for long-running jobs. When launching from the App, choose "Delegate" to run the task in the background via a FiftyOne orchestrator. Progress is tracked via `ctx.set_progress()`.

## Compatible models

Any VLM that vLLM can serve works with this plugin. Models tested or expected to work:

| Model family | Example model ID | Notes |
|-------------|-----------------|-------|
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | Best all-around; native 0-1000 grounding |
| Qwen3-VL | `Qwen/Qwen3-VL-4B` | |
| LLaVA | `llava-hf/llava-v1.6-mistral-7b-hf` | May need `--chat-template` on server |
| InternVL | `OpenGVLab/InternVL2_5-8B` | Use `pixel` coordinate format for detect |
| Pixtral | `mistralai/Pixtral-12B-2409` | |
| Phi-Vision | `microsoft/Phi-3.5-vision-instruct` | |
| Llama Vision | `meta-llama/Llama-3.2-11B-Vision-Instruct` | |
| Gemma | `google/gemma-3-4b-it` | |

Chat templates are handled automatically by vLLM. For models without a built-in template, configure it on the server with `vllm serve --chat-template <path>`.

## Error handling

- Per-sample errors are stored in `{field_name}_error` (e.g., `vllm_infer_caption_error`)
- A failed API call for one sample does not block the rest of the batch
- Connection is validated before processing begins (`engine.validate_connection()`)
- Detection post-validation silently drops malformed boxes (degenerate dimensions, wrong array length)

## Requirements

- Python >= 3.11
- FiftyOne >= 1.13.2
- `openai >= 1.0`
- `pillow >= 9.0`
- vLLM >= 0.16 (server-side, for `StructuredOutputsParams` support)

No GPU or CUDA dependencies on the client. All inference runs on the vLLM server.
