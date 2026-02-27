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

## Tasks

| Task | Output | FiftyOne Type | Description |
|------|--------|---------------|-------------|
| **Caption** | text | `StringField` | Generate image descriptions |
| **Classify** | single label | `fo.Classification` | Assign one class from a fixed set |
| **Tag** | multiple labels | `fo.Classifications` | Assign multiple labels from a fixed set |
| **Detect** | bounding boxes | `fo.Detections` | Locate objects with labels and boxes |
| **VQA** | text | `StringField` | Answer a question about the image |
| **OCR** | text | `StringField` | Extract visible text |
| **Custom** | text | `StringField` | User-defined prompt |

All tasks use vLLM structured output (`StructuredOutputsParams`) to guarantee valid, parseable responses at the token level. There is no regex or free-text parsing anywhere in the plugin.

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
        "output_field": "caption",
    },
    dataset_name=dataset.name,
)

# Results are stored directly on each sample
print(dataset.first().caption)
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
        "output_field": "scene_type",
    },
    dataset_name="my-images",
)

# fo.Classification label
print(dataset.first().scene_type.label)
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
        "output_field": "scene_tags",
    },
    dataset_name="my-images",
)

# fo.Classifications with multiple labels
for c in dataset.first().scene_tags.classifications:
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
        "output_field": "detections",
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
        "output_field": "vehicle_detections",
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
        "output_field": "people_count",
    },
    dataset_name="my-images",
)
```

### OCR

```python
foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "ocr",
        "output_field": "ocr_text",
    },
    dataset_name="my-images",
)
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
        "output_field": "moderation",
    },
    dataset_name="my-images",
)
```

### Run on a filtered view

```python
dataset = fo.load_dataset("my-images")
view = dataset.match(F("metadata.width") > 1000)

foo.execute_operator(
    "@fo-vllm/run_vllm_inference",
    params={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "task": "caption",
        "output_field": "caption",
    },
    dataset_name=dataset.name,
    # The operator processes ctx.target_view(), which
    # respects the current view set in the App or SDK
)
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
        "output_field": "color_caption",
    },
    dataset_name="my-images",
)
```

## Direct engine usage

For advanced workflows outside the operator, use the engine and task config directly:

```python
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
images = build_image_contents(filepaths, image_mode="base64")
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

### Advanced settings

Toggle "Show advanced settings" in the operator form to access these:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | task-specific | Sampling temperature (0.0 for deterministic tasks, 0.2 for generative) |
| `max_tokens` | 512 | Maximum tokens per response |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `batch_size` | 32 | Samples per inference batch |
| `max_concurrent` | 64 | Parallel HTTP requests to vLLM |
| `max_workers` | 8 | Threads for image loading/encoding |
| `image_mode` | auto | Image transfer method: `auto`, `base64`, or `filepath` |
| `coordinate_format` | normalized_1000 | Detection coordinate convention (detect task only) |

### Image mode

| Mode | When to use |
|------|-------------|
| `auto` | Default. Uses `filepath` for local servers, `base64` for remote. |
| `filepath` | Local vLLM server with `--allowed-local-media-path`. Zero I/O overhead. |
| `base64` | Remote servers or any setup. Images are base64-encoded in parallel. |

### Detection coordinate format

Different VLMs use different coordinate conventions for bounding boxes:

| Format | Range | Models |
|--------|-------|--------|
| `normalized_1000` | 0--1000 | Qwen2-VL, Qwen2.5-VL, Qwen3-VL (default) |
| `normalized_1` | 0.0--1.0 | Some fine-tuned models |
| `pixel` | 0--image_dim | InternVL, others |

The plugin converts all formats to FiftyOne's `[x, y, w, h]` relative coordinates automatically.

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

- Per-sample errors are stored in `{output_field}_error` (e.g., `caption_error`)
- A failed API call for one sample does not block the rest of the batch
- The operator UI warns if the output field already exists with an incompatible type
- Connection is validated before processing begins

## Requirements

- FiftyOne >= 0.25
- `openai >= 1.0`
- `pillow >= 9.0`

No GPU or CUDA dependencies. All inference runs on the vLLM server.
