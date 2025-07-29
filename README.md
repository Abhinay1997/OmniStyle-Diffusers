# OmniStyle-Diffusers
OmniStyle ported to diffusers pipeline

## Usage

This seems to work well only with square images. I have only tested it with 1024x1024 square images which gave good results and original resolutions which gave cropped and mixed up images. Feel free to raise an issue if you find bugs/inconsistencies.
```python
import torch
from PIL import Image
import os
from pipeline_omniflux import FluxPipeline

def crop_if_not_square(img):
    """Crops the input PIL image to a square by centering it."""
    w, h = img.size
    if w == h:
        return img
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))


## Load the pipeline and OmniStyle LoRA
from pipeline_omniflux import FluxPipeline
MODEL_ID = "black-forest-labs/FLUX.1-Dev"
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(
    "StyleXX/OmniStyle",
    weight_name="dit_lora.safetensors",
    adapter_name="omni"
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

## Instruction guided content style edit
output_dir = './'
content_path = "assets/bridge.png"
style_path = "assets/Pop Art_s0743____0907_01_query_0_img_000176_1683312941806_021301658999063.jpg.jpg"
output_filename = os.path.join(output_dir, "test_image_guided_output.jpg")

# Load and preprocess images
content_img_pil = Image.open(content_path).convert('RGB')
style_img_pil = Image.open(style_path).convert('RGB')

content_img_pil = crop_if_not_square(content_img_pil).resize((1024, 1024))
style_img_pil = crop_if_not_square(style_img_pil).resize((1024, 1024))

# The reference images are a list containing the style and content images
ref_imgs = [style_img_pil, content_img_pil]

# Run the pipeline
# Note: The prompt is empty for purely image-guided transfer
result_image = pipe(
    prompt="",
    ref_imgs=ref_imgs,
    num_inference_steps=25,
    guidance_scale=4.0,
    generator=torch.Generator('cuda').manual_seed(0),
    height=1024,
    width=1024,
    pe="d"  # Use diagonal positional encoding as in the original script. Other values are 'h', 'w', 'o'. h & w append along height or width and o is overlap.
).images[0]

result_image.save(output_filename)

## Image guided style transfer

content_path = "assets/tower.jpg"
instruction = "A vibrant and colorful painting in the style of pop art."
output_filename = os.path.join(output_dir, "test_instruction_guided_output.jpg")

# Load and preprocess the content image
content_img_pil = Image.open(content_path).convert("RGB")
content_img_pil = crop_if_not_square(content_img_pil).resize((1024, 1024))

# For instruction-guided transfer, only the content image is passed as a reference
ref_imgs = [content_img_pil]

# Run the pipeline
result_image = pipe(
    prompt=instruction,
    ref_imgs=ref_imgs,
    num_inference_steps=25,
    guidance_scale=4.0,
    height=1024,
    width=1024,
    generator=torch.Generator('cuda').manual_seed(0),
    pe="d"
).images[0]
result_image.save(output_filename)
```
