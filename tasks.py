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
            "system": (
                "You are an image captioner. Respond with a JSON object:"
                ' {"text": "your description"}'
            ),
            "prompt": "Describe this image concisely.",
            "output_type": "Classification",
            "default_field": "caption",
            "default_temperature": 0.2,
        },
        "classify": {
            "system": (
                "You are an image classifier. Respond with exactly one class label."
            ),
            "prompt": "Classify this image. Choose exactly one: {classes}",
            "output_type": "Classification",
            "default_field": "classification",
            "default_temperature": 0.0,
        },
        "tag": {
            "system": (
                "You are an image tagger. Respond with a JSON object:"
                ' {"labels": ["tag1", "tag2", ...]}'
            ),
            "prompt": ("Tag this image with all applicable labels from: {classes}"),
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
            "system": (
                "You are a visual question answerer. Respond with a JSON"
                ' object: {"answer": "your answer"}'
            ),
            "prompt": "{question}",
            "output_type": "Classification",
            "default_field": "vqa_answer",
            "default_temperature": 0.2,
        },
        "ocr": {
            "system": (
                "You are an OCR engine. Respond with a JSON object:"
                ' {"text": "extracted text"}'
            ),
            "prompt": "Extract all text visible in this image.",
            "output_type": "Classification",
            "default_field": "ocr_text",
            "default_temperature": 0.0,
        },
        "custom": {
            "system": ('Respond with a JSON object: {"response": "your response"}'),
            "prompt": "{prompt}",
            "output_type": "Classification",
            "default_field": "vlm_output",
            "default_temperature": 0.2,
        },
    }

    _COORD_FORMATS = {
        "normalized_1000": {
            "desc": (
                "0-1000 normalized coordinates where 0 is top-left"
                " and 1000 is bottom-right"
            ),
            "item_schema": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            },
        },
        "normalized_1": {
            "desc": (
                "0-1 normalized coordinates where 0.0 is top-left"
                " and 1.0 is bottom-right"
            ),
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
        coordinate_format="normalized_1000",
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

        if task == "detect":
            coord = self._COORD_FORMATS.get(
                coordinate_format, self._COORD_FORMATS["normalized_1000"]
            )
            coord_desc = coord["desc"]

            default_system = (
                "You are an object detector. Respond with a JSON object:"
                ' {"detections": [{"label": "...", "box": [x_min, y_min,'
                " x_max, y_max]}, ...]}. Use " + coord_desc + "."
            )
            default_prompt = "Detect all objects in this image."
            default_prompt_with_classes = (
                "Detect these objects in this image: {classes}."
                " For each object, return its label and bounding box"
                " as [x_min, y_min, x_max, y_max] in " + coord_desc + "."
            )
        else:
            default_system = defaults.get("system", "")
            default_prompt = defaults.get("prompt", "")
            default_prompt_with_classes = None

        self.system_prompt = (
            system_prompt if system_prompt is not None else default_system
        )

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
        "custom": "response",
    }

    def get_structured_outputs(self):
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
                            "items": {
                                "type": "string",
                                "enum": self.classes,
                            },
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

    def parse_response(self, text):
        """Parse VLM response into a FiftyOne label.

        All responses are structured: either JSON from json= constraint
        or a bare string from choice= constraint. json.loads() is the
        only parsing mechanism used.

        Every task returns a label type (Classification, Classifications,
        or Detections), enabling dynamic attributes for metadata.
        """
        if self.output_type == "Classification":
            if self.task in self._STRING_KEYS:
                data = json.loads(text)
                key = self._STRING_KEYS[self.task]
                return fo.Classification(label=data[key])
            return fo.Classification(label=text.strip())

        data = json.loads(text)

        if self.output_type == "Classifications":
            return fo.Classifications(
                classifications=[
                    fo.Classification(label=label) for label in data["labels"]
                ]
            )

        if self.output_type == "Detections":
            return self._parse_detections(data)

        raise ValueError(f"Unknown output type: {self.output_type}")

    def _parse_detections(self, data):
        """Post-generation validation for detection output.

        The JSON structure is guaranteed by the schema constraint.
        This method validates what schemas cannot enforce:
        coordinate ranges, degenerate boxes, array lengths.
        """
        detections = []
        raw = data.get("detections", [])

        for det in raw:
            box = det.get("box", [])
            if len(box) != 4:
                continue

            result = _convert_box(*[float(v) for v in box], self.coordinate_format)
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


_COORD_MAX = {"normalized_1000": 1000.0, "normalized_1": 1.0}


def _convert_box(x1, y1, x2, y2, coordinate_format):
    """Convert [x1,y1,x2,y2] to FiftyOne [x,y,w,h] in [0,1].

    Returns None if degenerate.
    """
    coord_max = _COORD_MAX.get(coordinate_format)
    if coord_max is not None:
        x1 = max(0.0, min(x1, coord_max))
        y1 = max(0.0, min(y1, coord_max))
        x2 = max(0.0, min(x2, coord_max))
        y2 = max(0.0, min(y2, coord_max))
        if x2 <= x1 or y2 <= y1:
            return None
        x = x1 / coord_max
        y = y1 / coord_max
        w = min((x2 - x1) / coord_max, 1.0 - x)
        h = min((y2 - y1) / coord_max, 1.0 - y)
    else:
        if x2 <= x1 or y2 <= y1:
            return None
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
    return x, y, w, h
