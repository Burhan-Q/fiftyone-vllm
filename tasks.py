"""TaskConfig: prompts, JSON schemas, output parsers, and post-generation
validation for all VLM inference tasks."""

import json
import logging

import fiftyone as fo

logger = logging.getLogger(__name__)


class TaskConfig:
    """Builds prompts, structured output constraints, and parses VLM
    responses."""

    TASKS = {
        "caption": {
            "system": ('You are an image captioner. Respond with a JSON object: {"text": "your description"}'),
            "prompt": "Describe this image concisely.",
            "output_type": "Classification",
            "default_field": "caption",
            "default_temperature": 0.2,
        },
        "classify": {
            "system": ("You are an image classifier. Respond with exactly one class label."),
            "system_open": ('You are an image classifier. Respond with a JSON object: {"label": "your label"}'),
            "prompt": "Classify this image. Choose exactly one: {classes}",
            "prompt_open": ("Classify this image with the single most appropriate label."),
            "output_type": "Classification",
            "default_field": "classification",
            "default_temperature": 0.0,
        },
        "tag": {
            "system": ('You are an image tagger. Respond with a JSON object: {"labels": ["tag1", "tag2", ...]}'),
            "prompt": ("Tag this image with all applicable labels from: {classes}"),
            "prompt_open": ("Tag this image with all applicable descriptive labels."),
            "output_type": "Classifications",
            "default_field": "tags",
            "default_temperature": 0.0,
        },
        "detect": {
            "output_type": "Detections",
            "default_field": "detections",
            "default_temperature": 0.0,
        },
        "vqa": {
            "system": ('You are a visual question answerer. Respond with a JSON object: {"answer": "your answer"}'),
            "prompt": "{question}",
            "output_type": "Classification",
            "default_field": "vqa_answer",
            "default_temperature": 0.2,
        },
        "ocr": {
            "system": ('You are an OCR engine. Respond with a JSON object: {"text": "extracted text"}'),
            "prompt": "Extract all text visible in this image.",
            "output_type": "Classification",
            "default_field": "ocr_text",
            "default_temperature": 0.0,
        },
    }

    _BOX_FORMATS = {
        "xyxy": {"labels": "[x_min, y_min, x_max, y_max]"},
        "xywh": {"labels": "[x, y, width, height]"},
        "cxcywh": {"labels": "[cx, cy, width, height]"},
    }

    _COORD_FORMATS = {
        "normalized_1000": {
            "desc": ("0-1000 normalized coordinates where 0 is top-left and 1000 is bottom-right"),
            "item_schema": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            },
        },
        "normalized_1": {
            "desc": ("0-1 normalized coordinates where 0.0 is top-left and 1.0 is bottom-right"),
            "item_schema": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "pixel": {
            "desc": "pixel coordinates",
            "item_schema": {"type": "number", "minimum": 0},
        },
    }

    def __init__(
        self,
        task,
        prompt=None,
        system_prompt=None,
        classes=None,
        coordinate_format="pixel",
        box_format="xyxy",
        **template_kwargs,
    ):
        """Initialize task configuration with prompts and constraints.

        Args:
            task: task identifier (one of TASKS keys).
            prompt: custom user prompt. If None, uses the task-specific
                default (or open-ended variant when classes is None).
            system_prompt: custom system prompt. If None, uses task default.
            classes: list of class labels for classify/tag/detect. If None,
                open-ended mode is used for classify/tag.
            coordinate_format: bounding box coordinate convention
                ("normalized_1000", "normalized_1", or "pixel").
            box_format: bounding box format ("xyxy", "xywh", or "cxcywh").
            **template_kwargs: additional format kwargs for prompt templates
                (e.g. question= for VQA).
        """
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Must be one of {list(self.TASKS)}")

        defaults = self.TASKS[task]
        self.task = task
        self.classes = classes
        self.output_type = defaults["output_type"]
        self.default_field = defaults["default_field"]
        self.default_temperature = defaults["default_temperature"]
        self.coordinate_format = coordinate_format
        self.box_format = box_format

        if task == "detect":
            coord = self._COORD_FORMATS.get(coordinate_format, self._COORD_FORMATS["normalized_1000"])
            coord_desc = coord["desc"]
            box_fmt = self._BOX_FORMATS.get(box_format, self._BOX_FORMATS["xyxy"])
            box_labels = box_fmt["labels"]

            default_system = (
                "You are an object detector. Respond with a JSON object:"
                ' {"detections": [{"label": "...", "box": ' + box_labels + "}, ...]}. Use " + coord_desc + "."
            )
            default_prompt = "Detect all objects in this image."
            default_prompt_with_classes = (
                "Detect these objects in this image: {classes}."
                " For each object, return its label and bounding box"
                " as " + box_labels + " in " + coord_desc + "."
            )
        else:
            open_ended = task in ("classify", "tag") and not classes
            default_system = defaults.get("system_open" if open_ended else "system", "")
            default_prompt = defaults.get("prompt_open" if open_ended else "prompt", "")
            default_prompt_with_classes = None

        self.system_prompt = system_prompt if system_prompt is not None else default_system

        if prompt is not None:
            raw_prompt = prompt
        elif task == "detect" and classes and default_prompt_with_classes:
            raw_prompt = default_prompt_with_classes
        else:
            raw_prompt = default_prompt

        fmt_kwargs = {**template_kwargs}
        if classes:
            fmt_kwargs["classes"] = ", ".join(classes)
        self.prompt = raw_prompt.format(**fmt_kwargs) if fmt_kwargs else raw_prompt

    def build_messages(self, image_content):
        """Build OpenAI-format messages for one image."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": self.prompt},
                ],
            }
        )
        return messages

    # -- Structured output constraints (vLLM 0.16+ StructuredOutputsParams) --

    _STRING_KEYS = {
        "caption": "text",
        "vqa": "answer",
        "ocr": "text",
    }

    def get_structured_outputs(self):
        """Return kwargs dict for StructuredOutputsParams.

        Every task returns a structured output constraint. Never returns None.

        Used as: extra_body={"structured_outputs": result}
        """
        if self.task == "classify":
            if self.classes:
                return {"choice": self.classes}
            return {
                "json": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "required": ["label"],
                    "additionalProperties": False,
                }
            }

        if self.task == "tag":
            items = {"type": "string", "enum": self.classes} if self.classes else {"type": "string"}
            return {
                "json": {
                    "type": "object",
                    "properties": {"labels": {"type": "array", "items": items}},
                    "required": ["labels"],
                    "additionalProperties": False,
                }
            }

        if self.task == "detect":
            label_schema = {"type": "string"}
            if self.classes:
                label_schema["enum"] = self.classes
            coord = self._COORD_FORMATS.get(
                self.coordinate_format,
                self._COORD_FORMATS["normalized_1000"],
            )
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
                                        "items": coord["item_schema"],
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

        if self.task in self._STRING_KEYS:
            key = self._STRING_KEYS[self.task]
            return {
                "json": {
                    "type": "object",
                    "properties": {key: {"type": "string"}},
                    "required": [key],
                    "additionalProperties": False,
                }
            }

        raise ValueError(f"No structured output schema for task: {self.task}")

    # -- Output parsing (all responses are structured) --

    def parse_response(self, text, image_width=None, image_height=None):
        """Parse VLM response into a FiftyOne label.

        All responses are structured: either JSON from json= constraint
        or a bare string from choice= constraint. json.loads() is the
        only parsing mechanism used.

        Every task returns a label type (Classification, Classifications,
        or Detections), enabling dynamic attributes for metadata.

        Args:
            text: raw VLM response string.
            image_width: pixel width of the source image (needed for
                pixel coordinate normalization in detection tasks).
            image_height: pixel height of the source image.
        """
        if self.output_type == "Classification":
            if self.task in self._STRING_KEYS:
                data = json.loads(text)
                key = self._STRING_KEYS[self.task]
                return fo.Classification(label=data[key])
            # classify: choice constraint returns bare text, json returns {"label": "..."}
            if self.task == "classify" and not self.classes:
                data = json.loads(text)
                return fo.Classification(label=data["label"])
            return fo.Classification(label=text.strip())

        data = json.loads(text)

        if self.output_type == "Classifications":
            return fo.Classifications(classifications=[fo.Classification(label=label) for label in data["labels"]])

        if self.output_type == "Detections":
            return self._parse_detections(data, image_width=image_width, image_height=image_height)

        raise ValueError(f"Unknown output type: {self.output_type}")

    def _parse_detections(self, data, image_width=None, image_height=None):
        """Post-generation validation for detection output.

        The JSON structure is guaranteed by the schema constraint.
        This method validates what schemas cannot enforce:
        coordinate ranges, degenerate boxes, array lengths.

        Args:
            data: parsed JSON dict with "detections" key.
            image_width: pixel width (required for pixel coordinate format).
            image_height: pixel height (required for pixel coordinate format).
        """
        detections = []
        raw = data.get("detections", [])

        for det in raw:
            box = det.get("box", [])
            if len(box) != 4:
                continue

            result = _convert_box(
                *[float(v) for v in box],
                coordinate_format=self.coordinate_format,
                box_format=self.box_format,
                img_w=image_width,
                img_h=image_height,
            )
            if result is None:
                continue

            detections.append(
                fo.Detection(
                    label=det.get("label", "object"),
                    bounding_box=list(result),
                )
            )

        if len(detections) < len(raw):
            logger.warning(
                "%d/%d detections dropped (bad length or degenerate box)",
                len(raw) - len(detections),
                len(raw),
            )

        return fo.Detections(detections=detections)


_COORD_SCALE = {"normalized_1000": 1000.0, "normalized_1": 1.0}


def _convert_box(v0, v1, v2, v3, coordinate_format, box_format="xyxy", img_w=None, img_h=None):
    """Convert model box output to FiftyOne [x, y, w, h] in [0, 1].

    Two-step pipeline:
      A. Convert from model box_format to internal xyxy.
      B. Normalize to [0, 1] based on coordinate_format.

    Returns None if degenerate or missing required dimensions.

    Args:
        v0, v1, v2, v3: box coordinates (meaning depends on box_format).
        coordinate_format: coordinate convention used by the model
            ("normalized_1000", "normalized_1", or "pixel").
        box_format: format of input coordinates ("xyxy", "xywh", "cxcywh").
        img_w: image width in pixels (required for "pixel" format).
        img_h: image height in pixels (required for "pixel" format).
    """
    # Step A: convert to xyxy
    if box_format == "xywh":
        x1, y1, x2, y2 = v0, v1, v0 + v2, v1 + v3
    elif box_format == "cxcywh":
        x1, y1, x2, y2 = v0 - v2 / 2, v1 - v3 / 2, v0 + v2 / 2, v1 + v3 / 2
    else:  # xyxy
        x1, y1, x2, y2 = v0, v1, v2, v3

    # Step B: resolve per-axis scale and normalize to [0, 1] xywh
    scale = _COORD_SCALE.get(coordinate_format)
    if scale is not None:
        max_x = max_y = scale
    elif coordinate_format == "pixel":
        if img_w is None or img_h is None:
            logger.warning("Pixel coordinates require image dimensions; skipping box")
            return None
        max_x, max_y = float(img_w), float(img_h)
    else:
        # Unknown format fallback — no normalization
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    x1 = max(0.0, min(x1, max_x))
    y1 = max(0.0, min(y1, max_y))
    x2 = max(0.0, min(x2, max_x))
    y2 = max(0.0, min(y2, max_y))
    if x2 <= x1 or y2 <= y1:
        return None
    x = x1 / max_x
    y = y1 / max_y
    w = min((x2 - x1) / max_x, 1.0 - x)
    h = min((y2 - y1) / max_y, 1.0 - y)
    return x, y, w, h
