import base64
from pathlib import Path

from openai import OpenAI

img = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
openai_api_key = "EMPTY"

client = OpenAI(api_key=openai_api_key, base_url="http://localhost:8811/v1")

# m = "Qwen/Qwen2.5-VL-3B-Instruct"  # not enough VRAM, may have been context tho
m = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"  # works with 8192 context

# Inference with direct HTTP URL
comp_res = client.chat.completions.create(
    model=m,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": img}},
            ],
        }
    ],
)

comp_res.choices[0].message.content
# 'The image depicts a serene landscape featuring a wooden boardwalk path that meanders through tall
# green grasses. The sky is clear with a few scattered clouds, and the overall scene is bathed in soft,
# natural light, suggesting it might be early morning or late afternoon. The boardwalk appears to be
# designed for walking and provides a pathway through the grassy area, likely leading to a more open
# space or another point of interest. The surrounding vegetation is lush and green, indicating a healthy,
# natural environment.'


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file content to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode("utf-8")
    return result


im_file = Path.cwd() / "bus.jpg"

b64_img = encode_base64_content_from_file(im_file)

b64_comp = client.chat.completions.create(
    model=m,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        }
    ],
)

b64_comp.choices[0].message.content
# 'The image shows a public transportation scene with a bus labeled "cero emisiones" (zero emissions)
#  in Madrid, Spain. The bus is electric and is part of the city\'s public transport system. There are
#  several people walking on the sidewalk near the bus. The background includes buildings with balconies
#  and greenery.'
