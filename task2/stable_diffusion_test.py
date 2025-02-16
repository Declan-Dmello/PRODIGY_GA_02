import torch
from diffusers import DiffusionPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")

prompt = "A sunset on mars"
image = pipe(prompt).images[0]
image.save("the_output1"
           ".png")