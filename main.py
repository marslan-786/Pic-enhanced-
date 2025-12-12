from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests
import io
from PIL import Image
import torch
import gc

app = FastAPI()

# Global Variable for the Heavy Model
upscaler_pipe = None

# --------------------------------------------
# LAZY LOAD: Stable Diffusion x4 Upscaler
# --------------------------------------------
def get_upscaler_model():
    global upscaler_pipe
    # Import inside function to avoid startup timeout
    from diffusers import StableDiffusionUpscalePipeline
    
    if upscaler_pipe is None:
        print("🚀 Loading Heavy Stable Diffusion Upscaler (VIP Mode)...")
        
        # Load the model directly to CPU (since we have 32GB RAM + 32 vCPU)
        upscaler_pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", 
            torch_dtype=torch.float32
        )
        upscaler_pipe = upscaler_pipe.to("cpu")
        
        # Memory Optimization (Optional since you have 32GB)
        # upscaler_pipe.enable_attention_slicing()
        
    return upscaler_pipe

# --------------------------------------------
# DOWNLOAD HELPER
# --------------------------------------------
def download_image(url):
    try:
        content = requests.get(url, timeout=10).content
        image = Image.open(io.BytesIO(content)).convert("RGB")
        return image
    except:
        raise HTTPException(400, "Invalid image URL")

# --------------------------------------------
# MAIN API
# --------------------------------------------
@app.get("/enhance")
def enhance(url: str, prompt: str = "high quality, 8k, ultra realistic, sharp focus"):
    
    # 1. Download Image
    low_res_img = download_image(url)
    
    # 2. Resize if too small (Model needs decent input) or too big
    # This model likes input around 128x128 or 256x256 to turn into 1024x1024
    if low_res_img.width > 512 or low_res_img.height > 512:
         low_res_img = low_res_img.resize((512, 512))

    print("⚡ Starting VIP Enhancement...")

    try:
        # 3. Get Model
        pipe = get_upscaler_model()

        # 4. Generate High Res Image
        # num_inference_steps=20 is good for speed/quality balance on CPU
        result = pipe(
            prompt=prompt,
            image=low_res_img,
            num_inference_steps=20,  
            guidance_scale=7.5
        ).images[0]

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    # 5. Return Image
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    # Clean up RAM slightly (optional)
    gc.collect()

    return StreamingResponse(buf, media_type="image/jpeg")

@app.get("/")
def home():
    return {"status": "VIP Enhancer Online", "ram": "Ready for Heavy Load"}

