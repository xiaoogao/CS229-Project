import torch
import pandas as pd
import os
import sys
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline

# 接收命令行参数：batch_id（0～3）
batch_id = int(sys.argv[1])
num_batches = 4

# 固定使用当前可见设备的 cuda:0
device = "cuda:0"
torch.cuda.set_device(0)

# 加载数据并划分
df = pd.read_csv("sdxl_prompts.csv")
total_len = len(df)
batch_size = total_len // num_batches

start_idx = batch_id * batch_size
end_idx = (batch_id + 1) * batch_size if batch_id < num_batches - 1 else total_len
df = df.iloc[start_idx:end_idx].reset_index(drop=True)

# 加载模型
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    
)
pipe.to(device)

negative_prompt = (
    "cartoon, illustration, anime, drawing, painting, sketch, abstract, deformed, distorted face, extra arms, blurry, "
    "bad anatomy, low quality, unrealistic, b&w, old photo, artistic style"
)

# 逐张生成并保存
for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Batch {batch_id}"):
    prompt = row["prompt"]
    filepath = row["filepath"]

    # ✅ 如果已经存在图像则跳过
    if os.path.exists(filepath):
        continue
    
    try:
        image = pipe(prompt=prompt,
                    # height=768, 
                    # width=768,
                    negative_prompt=negative_prompt,
                    guidance_scale=7.5).images[0]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Batch {batch_id}] Failed for: {prompt} — {e}")

