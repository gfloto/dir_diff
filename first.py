import torch
from diffusers import VQDiffusionPipeline

pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16, revision="fp16")
pipeline = pipeline.to("cuda")

image = pipeline("teddy bear playing in the pool").images[0]

# save image
image.save("./teddy_bear.png")
