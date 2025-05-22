import torch
import pandas as pd
import os
import sys
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline

# Parse batch_id from command-line arguments (should be 0~3)
batch_id = int(sys.argv[1])
num_batches = 4

# Set the CUDA device (change index if using CUDA_VISIBLE_DEVICES)
device = "cuda:0"
torch.cuda.set_device(0)

# Load the prompt CSV and split into batches for parallel generation
df = pd.read_csv("sdxl_prompts.csv")
total_len = len(df)
batch_size = total_len // num_batches

start_idx = batch_id * batch_size
end_idx = (batch_id + 1) * batch_size if batch_id < num_batches - 1 else total_len
df = df.iloc[start_idx:end_idx].reset_index(drop=True)

# Load the pretrained diffusion model (Stable Diffusion XL)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to(device)

# Set a negative prompt to filter out unwanted image styles
negative_prompt = (
    "cartoon, illustration, anime, drawing, painting, sketch, abstract, deformed, distorted face, extra arms, blurry, "
    "bad anatomy, low quality, unrealistic, b&w, old photo, artistic style"
)

# Generate and save images one by one
for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Batch {batch_id}"):
    prompt = row["prompt"]
    filepath = row["filepath"]

    # Skip generation if the image already exists
    if os.path.exists(filepath):
        continue

    try:
        image = pipe(
            prompt=prompt,
            # height=768,  # Uncomment to force resolution
            # width=768,
            negative_prompt=negative_prompt,
            guidance_scale=7.5
        ).images[0]
        image = image.resize((256, 256), Image.LANCZOS)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Batch {batch_id}] Failed for: {prompt} — {e}")
