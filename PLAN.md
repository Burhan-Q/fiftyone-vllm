# fo-vllm: FiftyOne + vLLM Plugin — Implementation Plan

## 1. Executive Summary

This plugin integrates vLLM with FiftyOne to provide universal VLM inference over image datasets via the **vLLM OpenAI-compatible API** (local or remote server). It supports **any model vLLM can serve** (Qwen2.5-VL, LLaVA, InternVL, Pixtral, Llama Vision, Phi-Vision, Gemma, etc.).

> **Scope**: This version targets online mode only. Offline mode (in-process GPU via `vllm.LLM`) is deferred. The architecture preserves extension points for adding it later (see Section 12).

**Key principles**:

1. **All tasks use structured output — no exceptions**: Every VLM response is constrained by `StructuredOutputsParams`. Tasks producing FiftyOne labels use `json=` schemas; classification uses `choice=`. Tasks producing text (caption, VQA, OCR, custom) also use `json=` schemas wrapping the text in a named key (e.g., `{"text": "..."}`). There is zero free-text parsing anywhere in the plugin. Post-generation validation catches what token-level constraints cannot enforce (numeric ranges, cross-field relationships).
2. **vLLM handles chat templates**: The plugin always sends OpenAI-format messages. vLLM automatically applies the correct chat template per model. Users configure custom chat templates on the vLLM server via `vllm serve --chat-template`.
3. **Operates on FiftyOne datasets directly**: The single operator works on `ctx.target_view()` (dataset, view, or selected samples) and writes results via `set_values()`.
4. **Parallel everything**: ThreadPool for image loading/encoding, async HTTP for API calls. Concurrency limits are user-configurable.

**Core value**: One plugin, any VLM, any task, any scale.

---

## 2. Design Evaluation

Three designs were evaluated. The chosen approach cherry-picks the best aspects of each.

### Design A: Thin Operator with Direct Loop

The operator directly manages iteration over samples and calls vLLM.

| Aspect | Assessment |
|--------|-----------|
| Simplicity | High — minimal abstraction |
| Reusability | Low — logic locked inside operator |
| Parallelism | Manual — must implement threading |

**Verdict**: Too monolithic. Mixing inference engine logic with FiftyOne operator logic hurts testability and reuse.

### Design B: FiftyOne Model Interface (`fo.Model`)

Implement FiftyOne's `Model` class and use `view.apply_model()`.

| Aspect | Assessment |
|--------|-----------|
| FiftyOne integration | Deep — works with apply_model, Model Zoo |
| Image handling | Wasteful — numpy decode/re-encode overhead |
| Prompt support | Awkward — Model interface has no prompt concept |
| Batch efficiency | Moderate — predict_all helps but images arrive as numpy |

**Verdict**: The `fo.Model` interface was designed for traditional vision models (input: tensor, output: label). VLMs need prompts, structured output constraints, and work best with file paths or base64 — not numpy arrays.

### Design C: Hybrid Engine + Smart Operator (CHOSEN)

Separate `VLLMEngine` for inference, `TaskConfig` for prompts/structured output/parsing, and an operator for FiftyOne integration.

| Aspect | Assessment |
|--------|-----------|
| Image handling | Optimal — base64 or file paths, no numpy |
| Batch efficiency | High — async concurrent API calls, bulk set_values() writes |
| Structured output | Full vLLM constrained generation (StructuredOutputsParams) |
| Separation of concerns | Clean — engine, task config, and operator each have one job |

### Cherry-Picked Elements

| From | Element | Reason |
|------|---------|--------|
| Design A | Direct control over batch loop | Avoids apply_model overhead for VLMs |
| Design B | Reusable class external to operator | Engine can be used via SDK |
| Design C | Parallel image prep + bulk writes | Maximum throughput |
| Design C | Smart image handling | base64 for remote, file:// paths for local |
| vLLM | `StructuredOutputsParams(choice=..., json=...)` | Enforced structured output at the token level |

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
 |               | + validation |                           |
 |               +------+-------+                           |
 |                      |                                   |
 |               +------v-------+                           |
 |               |  VLLMEngine  |                           |
 |               | AsyncOpenAI  |                           |
 |               | + semaphore  |                           |
 |               +--------------+                           |
 +---------------------------------------------------------+
```

### Image flow

```
Online (remote server):   filepath → ThreadPool base64 encode → image_url content → HTTP
Online (local server):    filepath → file:// URL string (no I/O) → image_url content → HTTP
```

The `file://` scheme is supported by the vLLM HTTP server (with `--allowed-local-media-path`). For remote servers, base64 encoding is the universally portable option.

---

## 4. File Structure

```
fo-vllm/
  fiftyone.yml              # Plugin manifest
  __init__.py               # register(plugin), operator imports
  engine.py                 # VLLMEngine: online inference via AsyncOpenAI
  tasks.py                  # TaskConfig: prompts, JSON schemas, output parsers + validation
  operators.py              # FiftyOne operator: UI, batching, progress, result storage
  utils.py                  # Image loading/encoding, async helpers
  requirements.txt          # Core deps (openai, pillow) — installed by FiftyOne plugin system
```

---

## 5. Task Taxonomy

The plugin supports 7 task types, selectable via a dropdown in the FiftyOne App. Each task defines three things: a **prompt template**, a **structured output constraint**, and an **output parser** (with post-generation validation where applicable).

### 5.1 Task Overview

| Task | Description | vLLM Constraint | Schema Key | FiftyOne Output Type | Parse Path |
|------|-------------|-----------------|------------|---------------------|------------|
| **Caption** | Generate image description | `json=` | `text` | `StringField` | `json.loads(r)["text"]` |
| **Classify** | Single-label classification | `choice=` | — | `fo.Classification` | `r.strip()` → label |
| **Tag** | Multi-label tagging | `json=` | `labels` | `fo.Classifications` | `json.loads(r)["labels"]` |
| **Detect** | Object detection with bboxes | `json=` | `detections` | `fo.Detections` | `json.loads(r)["detections"]` → validate |
| **VQA** | Visual question answering | `json=` | `answer` | `StringField` | `json.loads(r)["answer"]` |
| **OCR** | Extract text from image | `json=` | `text` | `StringField` | `json.loads(r)["text"]` |
| **Custom** | User-defined prompt | `json=` | `response` | `StringField` | `json.loads(r)["response"]` |

Every task uses structured output. Every parse path is `json.loads()` on schema-enforced JSON (or `.strip()` on a `choice=`-enforced string). There is no text parsing, no regex extraction, no fuzzy matching, no heuristic cleanup, anywhere in the plugin.

### 5.2 Per-Task Specifications

#### Caption

- **Default prompt**: `"Describe this image concisely."`
- **System prompt**: `"You are an image captioner. Respond with a JSON object: {\"text\": \"your description\"}"`
- **Structured output**: `StructuredOutputsParams(json=schema)` with schema:

```json
{
  "type": "object",
  "properties": {
    "text": {"type": "string"}
  },
  "required": ["text"],
  "additionalProperties": false
}
```

- **Output parse**: `json.loads(text)["text"]` → `StringField`
- **Default output field**: `caption`

#### Classify

- **Default prompt**: `"Classify this image. Choose exactly one: {classes}"`
- **System prompt**: `"You are an image classifier. Respond with exactly one class label."`
- **Structured output**: `StructuredOutputsParams(choice=classes)` — forces output to be exactly one of the class names
- **Output parse**: `fo.Classification(label=output_text.strip())`
- **Additional inputs**: `classes` (required, comma-separated list)
- **Default output field**: `classification`

#### Tag

- **Default prompt**: `"Tag this image with all applicable labels from: {classes}"`
- **System prompt**: `"You are an image tagger. Respond with a JSON object: {\"labels\": [\"tag1\", \"tag2\", ...]}"`
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
  "required": ["labels"],
  "additionalProperties": false
}
```

- **Output parse**: `json.loads(text)["labels"]` → `fo.Classifications(classifications=[fo.Classification(label=l) for l in labels])`
- **Additional inputs**: `classes` (required, comma-separated list)
- **Default output field**: `tags`

#### Detect

- **Default prompt**: `"Detect all objects in this image. For each object, return its label and bounding box as [x_min, y_min, x_max, y_max] in 0-1000 normalized coordinates."`
  (If classes provided: `"Detect these objects in this image: {classes}. ..."`)
- **System prompt**: `"You are an object detector. Respond with a JSON object: {\"detections\": [{\"label\": \"...\", \"box\": [x_min, y_min, x_max, y_max]}, ...]}. Use 0-1000 normalized coordinates where 0 is top-left and 1000 is bottom-right."`
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
        "required": ["label", "box"],
        "additionalProperties": false
      }
    }
  },
  "required": ["detections"],
  "additionalProperties": false
}
```

(When classes are provided, the `label` field adds `"enum": ["<class1>", ...]`)

- **Output parse**: JSON parse → **post-generation validation** (clamp coordinates to [0, 1000], reject degenerate boxes where x2 ≤ x1 or y2 ≤ y1, verify 4-element arrays) → convert `[x1, y1, x2, y2]` from 0-1000 to FiftyOne's `[x, y, w, h]` relative format → `fo.Detections`
- **Additional inputs**: `classes` (optional), `coordinate_format` (advanced, default `normalized_1000`)
- **Default output field**: `detections`

**Coordinate format note**: The 0-1000 normalized convention matches the Qwen2-VL/Qwen2.5-VL/Qwen3-VL model family's native grounding format. Other VLMs may use different coordinate systems. The `coordinate_format` advanced setting supports:

| Format | Range | Models | Conversion |
|--------|-------|--------|-----------|
| `normalized_1000` | 0–1000 | Qwen2-VL, Qwen2.5-VL, Qwen3-VL | ÷ 1000 |
| `normalized_1` | 0.0–1.0 | Some fine-tuned models | Direct use |
| `pixel` | 0–image_dim | InternVL, others | Requires `sample.metadata` |

Detection is inherently less "universal" than classification/captioning — accuracy and coordinate format depend heavily on the specific VLM's grounding capabilities.

**Schema backend note**: `minItems`/`maxItems` constraints are ignored by the default xgrammar backend. When vLLM's `auto` backend selector routes to guidance or outlines, they are enforced. The post-parse validation step handles the case when they are not.

#### VQA

- **Default prompt**: `"{question}"`
- **System prompt**: `"You are a visual question answerer. Respond with a JSON object: {\"answer\": \"your answer\"}"`
- **Structured output**: `StructuredOutputsParams(json=schema)` with schema:

```json
{
  "type": "object",
  "properties": {
    "answer": {"type": "string"}
  },
  "required": ["answer"],
  "additionalProperties": false
}
```

- **Output parse**: `json.loads(text)["answer"]` → `StringField`
- **Additional inputs**: `question` (required)
- **Default output field**: `vqa_answer`

#### OCR

- **Default prompt**: `"Extract all text visible in this image."`
- **System prompt**: `"You are an OCR engine. Respond with a JSON object: {\"text\": \"extracted text\"}"`
- **Structured output**: `StructuredOutputsParams(json=schema)` with schema:

```json
{
  "type": "object",
  "properties": {
    "text": {"type": "string"}
  },
  "required": ["text"],
  "additionalProperties": false
}
```

- **Output parse**: `json.loads(text)["text"]` → `StringField`
- **Default output field**: `ocr_text`

#### Custom

- **Default prompt**: User-provided (required)
- **System prompt**: `"Respond with a JSON object: {\"response\": \"your response\"}"` (user-overridable)
- **Structured output**: `StructuredOutputsParams(json=schema)` with schema:

```json
{
  "type": "object",
  "properties": {
    "response": {"type": "string"}
  },
  "required": ["response"],
  "additionalProperties": false
}
```

- **Output parse**: `json.loads(text)["response"]` → `StringField`
- **Default output field**: `vlm_output`

### 5.3 Structured Output Strategy

**Policy: All tasks use structured output. No exceptions.** Every VLM response is constrained by a JSON schema via `StructuredOutputsParams(json=...)`. The `choice=` constraint for classification is the sole non-JSON structured output type, retained because it is more efficient than a JSON wrapper for single-label selection and is itself a structured output mechanism.

This means:
- Every response from vLLM is either a valid JSON string (parseable via `json.loads()`) or a single token from a choice list
- There is zero text parsing anywhere in the plugin
- Every task has an explicit, declared schema
- Adding metadata fields to any task later (confidence, reasoning, etc.) is a schema change, not a parser change

**What structured output enforces** (reliably, across all backends):

| Constraint type | Mechanism | Example |
|----------------|-----------|---------|
| Single label from fixed set | `choice=` | Output is exactly `"cat"` or `"dog"` |
| JSON structure (keys, types, nesting) | `json=` schema | Output has `{"detections": [...]}` |
| String values from enum | `enum` in schema | Labels constrained to predefined classes |
| No extraneous fields | `additionalProperties: false` | Only declared keys appear in output |

**What structured output cannot enforce** (no backend supports at token level):

| Constraint type | Why not | Mitigation |
|----------------|---------|-----------|
| Numeric ranges (0 ≤ x ≤ 1000) | Regex/grammar can't encode value semantics | Post-parse clamping |
| Cross-field relationships (x2 > x1) | JSON Schema has no cross-field syntax | Post-parse rejection of degenerate boxes |
| Array length (`minItems`/`maxItems`) | xgrammar ignores; outlines/guidance support | Post-parse length validation |

**No fallback to free-text parsing.** If structured output fails for a sample (backend error, truncation, etc.), that sample errors into a `{field}_error` field. Silent degradation to unstructured output is not permitted — it would produce data that violates the parsing contract, which is worse than a visible error.

**Performance implications**: For string-output tasks (caption, VQA, OCR, custom), the JSON wrapper adds ~5-8 tokens of overhead per response (`{"text": "` prefix + `"}` suffix). Content quality inside the string value is unaffected. The guidance backend can actually *speed up* generation via jump-forward decoding on the deterministic structural tokens. For a 512-token response, the overhead is ~1.5%.

**System prompt hints**: Every task's system prompt includes the expected JSON structure. vLLM's documentation confirms best practice: "normally it's better to indicate in the prompt that a JSON needs to be generated and which fields and how should the LLM fill them. This can improve the results notably in most cases." The schema *enforces* structure regardless, but the prompt hint improves content quality.

**`additionalProperties: false`**: Every schema includes this. It prevents the model from adding unexpected fields, reduces wasted tokens, and ensures `json.loads()` results can be indexed directly without defensive checks.

**Engine mapping** (vLLM 0.16+ `StructuredOutputsParams` API):

- **Online mode**: `extra_body={"structured_outputs": {"choice": [...]}}` or `extra_body={"structured_outputs": {"json": schema}}`

---

## 6. Component Specifications

### 6.1 `VLLMEngine` (`engine.py`)

Thin wrapper over vLLM's OpenAI-compatible API with structured output.

```python
class VLLMEngine:
    """Thin wrapper over vLLM's OpenAI-compatible API with structured output."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 64,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        seed: int | None = None,
    ):
        from openai import AsyncOpenAI
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self._max_concurrent = max_concurrent
        self._aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def list_models(self) -> list[str]:
        """Query available models from the vLLM server."""
        from openai import OpenAI
        sync_client = OpenAI(
            base_url=self._aclient.base_url, api_key=self._aclient.api_key
        )
        return [m.id for m in sync_client.models.list().data]

    def validate_connection(self):
        """Test server connectivity. Raises on failure."""
        models = self.list_models()
        if not models:
            raise ConnectionError("vLLM server returned no models")

    def infer_batch(
        self,
        messages: list[list[dict]],
        structured_outputs: dict,
    ) -> list[str]:
        """Run batch inference with structured output constraints.

        Args:
            messages: list of OpenAI-format message lists, one per sample.
            structured_outputs: dict passed to vLLM's StructuredOutputsParams.
                Every task provides this — it is never None. Examples:
                  {"choice": ["cat", "dog"]}
                  {"json": {<JSON schema>}}

        Returns list of response strings (valid JSON or choice-constrained string).
        """
        return _run_async(self._async_infer_batch(messages, structured_outputs))

    async def _async_infer_batch(self, messages, structured_outputs) -> list[str]:
        extra_body = {"structured_outputs": structured_outputs}
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _call(msgs):
            async with sem:
                resp = await self._aclient.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    seed=self.seed,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content

        return list(await asyncio.gather(*[_call(m) for m in messages]))


import asyncio

def _run_async(coro):
    """Run an async coroutine safely, handling existing event loops.

    FiftyOne's App runs a Uvicorn server with its own event loop.
    Calling asyncio.run() from within that context raises
    RuntimeError: 'This event loop is already running'.
    This helper detects that case and runs in a dedicated thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
```

### 6.2 `TaskConfig` (`tasks.py`)

Handles prompt construction, structured output schema generation, and deterministic output parsing per task type — including post-generation validation for detection.

```python
import json
import fiftyone as fo


class TaskConfig:
    """Builds prompts, structured output constraints, and parses VLM responses."""

    TASKS = {
        "caption": {
            "system": "You are an image captioner. Respond with a JSON object: {\"text\": \"your description\"}",
            "prompt": "Describe this image concisely.",
            "output_type": "string",
            "default_field": "caption",
            "default_temperature": 0.2,
        },
        "classify": {
            "system": "You are an image classifier. Respond with exactly one class label.",
            "prompt": "Classify this image. Choose exactly one: {classes}",
            "output_type": "Classification",
            "default_field": "classification",
            "default_temperature": 0.0,
        },
        "tag": {
            "system": "You are an image tagger. Respond with a JSON object: {\"labels\": [\"tag1\", \"tag2\", ...]}",
            "prompt": "Tag this image with all applicable labels from: {classes}",
            "output_type": "Classifications",
            "default_field": "tags",
            "default_temperature": 0.0,
        },
        "detect": {
            "system": (
                "You are an object detector. Respond with a JSON object: "
                "{\"detections\": [{\"label\": \"...\", \"box\": [x_min, y_min, x_max, y_max]}, ...]}. "
                "Use 0-1000 normalized coordinates where 0 is top-left and 1000 is bottom-right."
            ),
            "prompt": "Detect all objects in this image.",
            "output_type": "Detections",
            "default_field": "detections",
            "default_temperature": 0.0,
        },
        "vqa": {
            "system": "You are a visual question answerer. Respond with a JSON object: {\"answer\": \"your answer\"}",
            "prompt": "{question}",
            "output_type": "string",
            "default_field": "vqa_answer",
            "default_temperature": 0.2,
        },
        "ocr": {
            "system": "You are an OCR engine. Respond with a JSON object: {\"text\": \"extracted text\"}",
            "prompt": "Extract all text visible in this image.",
            "output_type": "string",
            "default_field": "ocr_text",
            "default_temperature": 0.0,
        },
        "custom": {
            "system": "Respond with a JSON object: {\"response\": \"your response\"}",
            "prompt": "{prompt}",
            "output_type": "string",
            "default_field": "vlm_output",
            "default_temperature": 0.2,
        },
    }

    def __init__(
        self,
        task: str,
        prompt: str | None = None,
        system_prompt: str | None = None,
        classes: list[str] | None = None,
        coordinate_format: str = "normalized_1000",
        **template_kwargs,
    ):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Must be one of {list(self.TASKS)}")

        defaults = self.TASKS[task]
        self.task = task
        self.classes = classes
        self.output_type = defaults["output_type"]
        self.default_field = defaults["default_field"]
        self.default_temperature = defaults["default_temperature"]
        self.coordinate_format = coordinate_format
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

    # Map of task → JSON key for string-output tasks
    _STRING_KEYS = {
        "caption": "text",
        "vqa": "answer",
        "ocr": "text",
        "custom": "response",
    }

    def get_structured_outputs(self) -> dict:
        """Return kwargs dict for StructuredOutputsParams.

        Every task returns a structured output constraint. Never returns None.

        Used as: extra_body={"structured_outputs": result}
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
                    "additionalProperties": False,
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
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["detections"],
                    "additionalProperties": False,
                }
            }

        if self.task == "vqa":
            return {
                "json": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                }
            }

        if self.task in ("caption", "ocr"):
            return {
                "json": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                }
            }

        if self.task == "custom":
            return {
                "json": {
                    "type": "object",
                    "properties": {"response": {"type": "string"}},
                    "required": ["response"],
                    "additionalProperties": False,
                }
            }

        raise ValueError(f"No structured output schema for task: {self.task}")

    # -- Output parsing (all responses are structured) --

    def parse_response(self, text: str) -> fo.Classification | fo.Classifications | fo.Detections | str:
        """Parse VLM response into a FiftyOne label or string.

        All responses are structured: either JSON from json= constraint
        or a bare string from choice= constraint. json.loads() is the
        only parsing mechanism used.
        """
        # Classify: choice= constraint, bare string output
        if self.output_type == "Classification":
            return fo.Classification(label=text.strip())

        # All other tasks: JSON from json= constraint
        data = json.loads(text)

        # String-output tasks (caption, vqa, ocr, custom)
        if self.output_type == "string":
            key = self._STRING_KEYS[self.task]
            return data[key]

        # Tag: array of labels
        if self.output_type == "Classifications":
            return fo.Classifications(
                classifications=[
                    fo.Classification(label=label)
                    for label in data["labels"]
                ]
            )

        # Detect: array of {label, box} with post-generation validation
        if self.output_type == "Detections":
            return self._parse_detections(data)

        raise ValueError(f"Unknown output type: {self.output_type}")

    def _parse_detections(self, data: dict) -> fo.Detections:
        """Post-generation validation for detection output.

        The JSON structure is guaranteed by the schema constraint.
        This method validates what schemas cannot enforce:
        coordinate ranges, degenerate boxes, array lengths.
        """
        detections = []

        # Determine coordinate scale based on format
        if self.coordinate_format == "normalized_1000":
            coord_max = 1000.0
        elif self.coordinate_format == "normalized_1":
            coord_max = 1.0
        else:
            # pixel mode: no clamping max, division handled differently
            coord_max = None

        for det in data.get("detections", []):
            box = det.get("box", [])

            # Validate array length (minItems/maxItems may not be enforced by xgrammar)
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]

            if coord_max is not None:
                # Clamp to valid coordinate range
                x1 = max(0.0, min(x1, coord_max))
                y1 = max(0.0, min(y1, coord_max))
                x2 = max(0.0, min(x2, coord_max))
                y2 = max(0.0, min(y2, coord_max))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Convert to FiftyOne [x, y, w, h] relative format (0-1)
                x = x1 / coord_max
                y = y1 / coord_max
                w = min((x2 - x1) / coord_max, 1.0 - x)
                h = min((y2 - y1) / coord_max, 1.0 - y)
            else:
                # Pixel mode: cannot convert without image dimensions
                # Store as-is; caller must provide metadata for conversion
                # This path is a placeholder for future per-sample metadata support
                if x2 <= x1 or y2 <= y1:
                    continue
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

            label = det.get("label", "object")

            detections.append(fo.Detection(
                label=label,
                bounding_box=[x, y, w, h],
            ))

        return fo.Detections(detections=detections)
```

### 6.3 Image Content Builder (`utils.py`)

Parallel construction of image content objects from file paths.

```python
import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor


def build_image_contents(
    filepaths: list[str],
    image_mode: str = "auto",
    base_url: str | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Build image content dicts for vLLM chat messages.

    Returns OpenAI-format image_url content dicts. All I/O-bound
    work is parallelized via ThreadPoolExecutor.

    Args:
        filepaths: absolute paths to image files.
        image_mode: "auto" | "base64" | "filepath"
            - "auto": filepath for local servers; base64 for remote
            - "filepath": file:// URL strings (local servers only, no I/O)
            - "base64": base64 data URIs (works everywhere, parallel I/O)
        base_url: vLLM server URL, used by "auto" to detect local servers.
        max_workers: ThreadPoolExecutor size for parallel encoding.
    """
    resolved = _resolve_image_mode(image_mode, base_url)

    if resolved == "filepath":
        # No I/O — instant
        return [_filepath_content(fp) for fp in filepaths]
    elif resolved == "base64":
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_encode_base64, filepaths))
    else:
        raise ValueError(f"Unknown image mode: {resolved}")


def _resolve_image_mode(image_mode, base_url):
    """Resolve the effective image mode from user settings and context."""
    if image_mode != "auto":
        return image_mode
    # Auto-detection: local server can use file paths
    if base_url and any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        return "filepath"
    return "base64"


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
```

### 6.4 Operator (`operators.py`)

Single operator for all VLM inference tasks. Uses FiftyOne's `types.*` API for a dynamic form.

```python
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

        return types.Property(inputs, view=types.View(label="vLLM Inference"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        params = ctx.params

        # 1. Resolve secrets with precedence: UI param > FiftyOne secret > env > default
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
            temperature=params.get("temperature", None),  # resolved below
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
            coordinate_format=params.get("coordinate_format", "normalized_1000"),
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

        # Get structured output constraints from task
        structured_outputs = task.get_structured_outputs()

        # 6. Process in batches
        processed = 0
        total_errors = 0

        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_paths = filepaths[i:i + batch_size]

            # 6a. Parallel image content construction
            image_contents = build_image_contents(
                batch_paths,
                image_mode=params.get("image_mode", "auto"),
                base_url=base_url,
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
            for sid, resp in zip(batch_ids, responses):
                try:
                    results[sid] = task.parse_response(resp)
                except Exception as e:
                    errors[sid] = f"{type(e).__name__}: {e}"
                    total_errors += 1

            # 6e. Bulk-write results and errors
            if results:
                ctx.dataset.set_values(output_field, results, key_field="id")
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
        return types.Property(outputs, view=types.View(label="Complete"))
```

---

## 7. Operator UI Specification

### 7.1 `_model_selector(ctx, inputs)`

```python
def _model_selector(ctx, inputs):
    inputs.str(
        "model",
        label="Model",
        required=True,
        description="HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)",
    )
```

### 7.2 `_server_settings(ctx, inputs)`

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

### 7.3 `_task_selector(ctx, inputs)`

```python
def _task_selector(ctx, inputs):
    task_dropdown = types.Dropdown(label="Task")
    task_dropdown.add_choice("caption", label="Caption",
        description="Generate a text description of the image")
    task_dropdown.add_choice("classify", label="Classify",
        description="Assign a single class label (constrained output)")
    task_dropdown.add_choice("tag", label="Tag",
        description="Assign multiple labels (constrained JSON output)")
    task_dropdown.add_choice("detect", label="Detect",
        description="Detect objects with bounding boxes (constrained JSON output)")
    task_dropdown.add_choice("vqa", label="VQA",
        description="Answer a question about the image")
    task_dropdown.add_choice("ocr", label="OCR",
        description="Extract text visible in the image")
    task_dropdown.add_choice("custom", label="Custom",
        description="Custom prompt with free-form response")
    inputs.enum(
        "task",
        task_dropdown.values(),
        required=True,
        label="Task",
        view=task_dropdown,
    )
    return ctx.params.get("task", None)
```

### 7.4 `_task_settings(ctx, inputs, task)`

```python
def _task_settings(ctx, inputs, task):
    if task is None:
        return

    inputs.view("task_header", types.Header(label="Task Settings", divider=True))

    # Classes input (classify, tag, detect)
    if task in ("classify", "tag"):
        inputs.str("classes", label="Classes", required=True,
            description="Comma-separated class names (e.g., cat, dog, bird)")
    elif task == "detect":
        inputs.str("classes", label="Classes", required=False,
            description="Optional: comma-separated classes to detect (leave empty for open detection)")

    # Question input (VQA)
    if task == "vqa":
        inputs.str("question", label="Question", required=True,
            description="Question to ask about each image")

    # Custom prompt input
    if task == "custom":
        inputs.str("prompt", label="Prompt", required=True,
            description="Prompt to send with each image")
        inputs.str("system_prompt", label="System Prompt", required=False,
            description="Optional system prompt for context")

    # Prompt override for non-custom tasks
    if task not in ("custom",):
        inputs.str("prompt_override", label="Prompt Override", required=False,
            description="Override the default prompt for this task")
```

### 7.5 `_output_settings(ctx, inputs, task)`

Each task has its own default field name to prevent type conflicts.

```python
def _output_settings(ctx, inputs, task):
    defaults = {t: v["default_field"] for t, v in TaskConfig.TASKS.items()}
    inputs.str(
        "output_field",
        label="Output Field",
        required=True,
        default=defaults.get(task, "vlm_output"),
        description="Field name to store results on each sample",
    )
```

### 7.6 `_field_conflict_check(ctx, inputs, task)`

Warn early if the output field already exists with an incompatible type.

```python
def _field_conflict_check(ctx, inputs, task):
    """Check for field type conflicts and warn in the UI."""
    output_field = ctx.params.get("output_field")
    if not output_field or not ctx.dataset:
        return

    existing = ctx.dataset.get_field(output_field)
    if existing is None:
        return  # Field doesn't exist yet, no conflict

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
                label=f"Field '{output_field}' already exists with type "
                      f"{type(existing).__name__}. This task writes "
                      f"{expected.__name__}. Choose a different field name "
                      f"to avoid conflicts."
            ),
        )
```

### 7.7 `_advanced_settings(ctx, inputs)`

```python
def _advanced_settings(ctx, inputs):
    inputs.view("adv_header", types.Header(label="Advanced Settings", divider=True))
    inputs.bool(
        "show_advanced", label="Show advanced settings",
        default=False, view=types.SwitchView(),
    )
    if not ctx.params.get("show_advanced", False):
        return

    # -- Sampling parameters --
    inputs.float("temperature", label="Temperature",
        default=None, min=0.0, max=2.0,
        description="Sampling temperature (leave empty for task-specific default)")
    inputs.int("max_tokens", label="Max Tokens",
        default=512, min=1, max=4096,
        description="Maximum tokens to generate per sample")
    inputs.float("top_p", label="Top P", default=1.0, min=0.0, max=1.0)
    inputs.int("batch_size", label="Batch Size",
        default=32, min=1, max=512,
        description="Number of samples per inference batch")

    # -- Parallelism controls --
    inputs.int("max_concurrent", label="Max Concurrent Requests",
        default=64, min=1, max=256,
        description="Maximum parallel HTTP requests to vLLM server")
    inputs.int("max_workers", label="Image Loading Workers",
        default=8, min=1, max=32,
        description="Thread pool size for parallel image loading/encoding")

    # -- Image handling --
    image_dropdown = types.Dropdown()
    image_dropdown.add_choice("auto", label="Auto",
        description="Best option based on server location")
    image_dropdown.add_choice("base64", label="Base64",
        description="Base64-encoded (works everywhere)")
    image_dropdown.add_choice("filepath", label="File Path",
        description="file:// paths (local servers with --allowed-local-media-path)")
    inputs.enum("image_mode", image_dropdown.values(),
        default="auto", label="Image Mode", view=image_dropdown)

    # -- Detection coordinate format --
    task = ctx.params.get("task")
    if task == "detect":
        coord_dropdown = types.Dropdown()
        coord_dropdown.add_choice("normalized_1000", label="0-1000 (Qwen default)")
        coord_dropdown.add_choice("normalized_1", label="0-1 (relative)")
        coord_dropdown.add_choice("pixel", label="Pixel coordinates")
        inputs.enum("coordinate_format", coord_dropdown.values(),
            default="normalized_1000", label="Coordinate Format",
            view=coord_dropdown,
            description="Bounding box coordinate convention used by the model")
```

---

## 8. Critical Technical Decisions

### 8.1 Chat Templates Are Handled by vLLM

The `/v1/chat/completions` endpoint automatically applies the correct Jinja2 chat template per model. The plugin always uses the universal OpenAI message format. For models without a built-in template (e.g., LLaVA-1.5), configure the chat template on the vLLM server via `vllm serve --chat-template <path>`. This is a server-side concern, not a plugin concern.

### 8.2 Structured Output is Universal

| Task | Constraint | What's guaranteed | What requires post-validation |
|------|-----------|-------------------|-------------------------------|
| Classify | `choice=classes` | Output is exactly one class name | Nothing |
| Tag | `json=schema` with enum | Valid JSON with enum-constrained labels | Nothing |
| Detect | `json=schema` | Valid JSON structure | Box length (xgrammar may ignore minItems/maxItems), coordinate ranges, degenerate boxes |
| Caption | `json={"text": string}` | Valid JSON wrapping text | Nothing |
| VQA | `json={"answer": string}` | Valid JSON wrapping answer | Nothing |
| OCR | `json={"text": string}` | Valid JSON wrapping text | Nothing |
| Custom | `json={"response": string}` | Valid JSON wrapping response | Nothing |

### 8.3 Image Transfer Strategy

| Scenario | Strategy | Content type | Why |
|----------|----------|-------------|-----|
| Local server | `file://` paths | `image_url` | Zero I/O; requires `--allowed-local-media-path` on server |
| Remote server | Parallel base64 | `image_url` | Only universally portable option |

### 8.4 Bulk Writes via `set_values()`

`dataset.set_values(field, {id: value, ...}, key_field="id")` performs a single bulk MongoDB write per batch. Dramatically faster than per-sample `save()`.

### 8.5 Async Concurrency

`AsyncOpenAI` with `asyncio.Semaphore(max_concurrent)` wrapped in `_run_async()` for event-loop safety. All requests in a batch are fired concurrently. The semaphore prevents overwhelming the server. vLLM's continuous batching handles the rest. The `_run_async()` helper detects existing event loops (e.g., FiftyOne's App server) and runs async code in a dedicated thread to avoid `RuntimeError`.

### 8.6 Detection Coordinate Convention

Configurable via `coordinate_format` advanced setting. Default is `normalized_1000` matching Qwen2-VL/Qwen2.5-VL/Qwen3-VL's native grounding format. Detection accuracy and coordinate format vary significantly across model families — this task is not "universal" in the way classification and captioning are. The system prompt reflects the chosen coordinate convention.

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
engine = VLLMEngine(model="Qwen/Qwen2.5-VL-7B-Instruct")
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
    try:
        results[sid] = task.parse_response(r)
    except Exception:
        pass  # handle errors

dataset.set_values("scene_type", results, key_field="id")
```

---

## 10. Implementation Plan

### Phase 1: Core Components (Parallelizable: 2 tracks)

**Track A: `engine.py` + `utils.py`**

1. `VLLMEngine.__init__()` — AsyncOpenAI client setup, sampling params
2. `_async_infer_batch()` with semaphore + event-loop-safe `_run_async()`
3. `list_models()` and `validate_connection()`
4. `build_image_contents()` — filepath mode (string formatting, no I/O)
5. `build_image_contents()` — base64 mode (ThreadPoolExecutor parallel I/O)
6. `build_image_contents()` — auto-detection logic

**Track B: `tasks.py`**

1. `TaskConfig.__init__()` with 7 task defaults (all with JSON system prompts), prompt template formatting, coordinate_format
2. `build_messages()` — builds OpenAI-format message list with image content
3. `get_structured_outputs()` — returns `dict` (never `None`) for all 7 tasks: choice for classify, json schemas for all others
4. `parse_response()` for string tasks (caption, vqa, ocr, custom — `json.loads()` → extract named key)
5. `parse_response()` for Classification (classify — direct label from structured choice)
6. `parse_response()` for Classifications (tag — JSON parse to fo.Classifications)
7. `_parse_detections()` — receives parsed dict, applies **post-generation validation** (coordinate clamping, degenerate box filtering, array length checks, configurable coordinate format)

### Phase 2: FiftyOne Integration

**Track C: `operators.py` + `__init__.py` + `fiftyone.yml`** (depends on Phase 1)

1. `fiftyone.yml` manifest
2. `_model_selector()` — text input for model ID
3. `_server_settings()` — base_url and api_key fields
4. `_task_selector()` — Dropdown with 7 task choices
5. `_task_settings()` — conditional fields per task
6. `_output_settings()` — task-specific default field names
7. `_field_conflict_check()` — early warning for field type conflicts
8. `_advanced_settings()` — sampling params, concurrency controls, image mode, coordinate format
9. `execute()` — connection validation, batch loop, image prep, inference, per-sample error handling, set_values, progress
10. `resolve_delegation()` and `resolve_output()`
11. `__init__.py` with `register()` function
12. `requirements.txt`

### Phase 3: Testing

1. Classify with `structured_outputs(choice=)` against a vLLM server
2. Tag with `structured_outputs(json=)`
3. Caption with `structured_outputs(json={"text": string})` — verify JSON wrapper
4. Detect with `structured_outputs(json=)` + coordinate validation
5. VQA with `structured_outputs(json={"answer": string})` — verify JSON wrapper
6. OCR with `structured_outputs(json={"text": string})` — verify JSON wrapper
7. Custom with `structured_outputs(json={"response": string})`
8. Progress reporting (immediate and delegated)
9. Image modes: base64, filepath (verify correct resolution)
10. Per-sample error handling: corrupted image in batch, truncated JSON response
11. Event loop safety: called from within FiftyOne App server
12. Field conflict detection: re-run with incompatible output field
13. Edge cases: empty dataset, missing files, model errors, max_tokens too low for JSON

### Parallelization Summary

```
Phase 1:  Track A (engine + utils) ──────────┐
          Track B (tasks)           ──────────┤
                                              v
Phase 2:  Track C (operator + plugin wiring) ─┐
                                               v
Phase 3:  Testing ─────────────────────────────>
```

---

## 11. Plugin Manifest + Requirements

### `fiftyone.yml`

```yaml
name: "@fo-vllm/vllm"
type: plugin
author: "fo-vllm contributors"
version: "0.1.0"
description: "Universal VLM inference via vLLM — any model, any task, any scale"
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
pillow>=9.0
```

No GPU/CUDA dependencies — all inference runs on the vLLM server.

---

## 12. Future Considerations

| Feature | How Architecture Supports It |
|---------|------------------------------|
| **Offline mode (in-process GPU)** | Add `mode` param to VLLMEngine, `_offline_batch()`, `cleanup()`. See extension points below |
| **Structured JSON extraction** | Add `"json_extract"` task to TaskConfig with user-provided schema → `structured_outputs(json=)` |
| **Multi-image per sample** | Extend `build_messages()` to accept multiple images per sample |
| **Video understanding** | Add video frame extraction to utils; some VLMs support video inputs |
| **Per-sample prompts from field** | Read prompt per-sample from a dataset field (e.g., `sample["question"]`) |
| **LoRA adapter selection** | vLLM supports runtime LoRA loading; add adapter parameter to engine |
| **Rate limiting** | Add requests-per-second throttle for external API endpoints (OpenAI, Together AI, etc.) |
| **Model comparison** | Run multiple models and store results in different fields |
| **Embeddings extraction** | Add embedding extraction mode for VLMs that expose embeddings |
| **FiftyOne Model Zoo** | Wrap VLLMEngine in `fo.Model` for `foz.load_zoo_model()` compat |
| **Panel UI** | React panel for interactive VLM chat with selected images |
| **Streaming responses** | vLLM streaming for interactive panel use |
| **Keypoint estimation** | Add `"keypoint"` task → `fo.Keypoints` |
| **Segmentation polygons** | Add `"segment"` task → `fo.Polylines` |
| **Model auto-discovery** | Query `/v1/models` endpoint to populate model dropdown dynamically |

### Extension Points for Future Offline Mode

Adding offline mode (in-process GPU via `vllm.LLM`) requires changes to 3 of the 4 source files:

1. **engine.py**: Add `mode` parameter to `__init__()`. Add offline-only params (`tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `dtype`, `trust_remote_code`, `enforce_eager`, `limit_mm_per_prompt`, `device`, `chat_template`). Add `_offline_batch()` using `LLM.chat()` with `SamplingParams(structured_outputs=StructuredOutputsParams(...))`. Add `cleanup()` for GPU memory freeing (`del self._llm` + `torch.cuda.empty_cache()`). Dispatch in `infer_batch()` based on mode.
2. **utils.py**: Add `mode` parameter to `build_image_contents()`. Add `pil` image mode using `PIL.Image.open()` (lazy-loaded; vLLM decodes pixels internally). Update `_resolve_image_mode()` to default to `pil` for offline mode and override `filepath→pil` since `file://` is not supported by `LLM.chat()`.
3. **operators.py**: Add `_mode_selector()` RadioGroup for online/offline. Make server settings conditional on online mode. Add offline engine settings (`tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, chat template, `_device_selector()`). Add `try/finally` with `engine.cleanup()` in `execute()`. Add vllm version check in UI.
4. **tasks.py**: No changes needed — task prompts, schemas, and parsers are mode-agnostic.
5. Add `requirements-local.txt` with `vllm>=0.16.0` for users who want offline mode.

### Architectural hooks

1. **TaskConfig is extensible**: New tasks add an entry to `TASKS`, a `get_structured_outputs()` branch, and a `parse_response()` branch. Engine and operator are untouched.
2. **VLLMEngine is API-agnostic**: New vLLM features (LoRA, speculative decoding, quantization) only touch `engine.py`.
3. **Operator UI is dynamic**: New parameters only require changes to the relevant helper function in `operators.py`.

---

## 13. Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Server unreachable | `validate_connection()` before batch processing; query `/v1/models` |
| Slow base64 encoding | ThreadPool parallelism with configurable `max_workers` |
| `max_tokens` too low for JSON output | Affects all tasks (all use JSON schemas). Minimum recommended: 32 for classify, 64 for caption/vqa/ocr, 128 for tag, 256 for detect. Validate in resolve_input |
| Structured output fails for a sample | No fallback to free-text parsing. Sample errors into `{field}_error` field. Silent degradation would produce unparseable data that violates the contract |
| Detection coordinates out of range | Post-parse clamping to valid range; reject degenerate boxes |
| `minItems`/`maxItems` ignored by xgrammar | Post-parse array length validation; schema kept for backends that support it |
| Non-Qwen model for detection | Configurable `coordinate_format`; document model compatibility |
| Individual sample parse failure | Per-sample try/except; errors stored in `{field}_error` field |
| Event loop already running (FiftyOne App) | `_run_async()` detects existing loop, runs async code in dedicated thread |
| Output field type conflict | `_field_conflict_check()` warns in UI before execution |
| Dataset too large for memory | Stream IDs/filepaths with batch iteration; never load all samples at once |
| Model lacks chat template | Configure on vLLM server via `vllm serve --chat-template <path>` |
