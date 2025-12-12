from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests
import io
from PIL import Image, ImageFilter
import numpy as np
import torch
import gc

app = FastAPI()

# Global variables (Models abhi load nahi honge)
gfpgan_model = None
sd_pipe = None

# --- MODELS KO FUNCTION KE ANDAR LOAD KAREN ---

def get_gfpgan():
    global gfpgan_model
    # Import yahan karen takay shuru mein time na lagay
    from gfpgan import GFPGANer
    if gfpgan_model is None:
        print("Loading GFPGAN Model for the first time...")
        gfpgan_model = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=2,
            arch='clean',
            channel_multiplier=2
        )
    return gfpgan_model

def get_sd_pipe():
    global sd_pipe
    from diffusers import StableDiffusionImg2ImgPipeline
    if sd_pipe is None:
        print("Loading Stable Diffusion (Heavy)...")
        # CPU use kar rahe hain to float32 theek hai
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32
        )
        sd_pipe = sd_pipe.to("cpu")
        sd_pipe.enable_attention_slicing() # Memory optimize
    return sd_pipe

def esrgan_process(img):
    from realesrgan import RealESRGAN
    device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x4plus.pth')
    return model.predict(img)

# --- API ROUTES ---

@app.get("/enhance")
def enhance(type: str, url: str):
    # Image download
    try:
        content = requests.get(url, timeout=10).content
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid Image URL")

    type = type.upper()
    output = None

    if type == "A":
        # GFPGAN
        model = get_gfpgan()
        _, _, restored = model.enhance(np.array(image), has_aligned=False)
        restored_pil = Image.fromarray(restored)
        output = esrgan_process(restored_pil)

    elif type == "B":
        # Stable Diffusion
        pipe = get_sd_pipe()
        # Steps kam kiye hain speed ke liye
        output = pipe(
            prompt="high quality, detailed", 
            image=image, 
            strength=0.75, 
            guidance_scale=7.5,
            num_inference_steps=15
        ).images[0]

    elif type == "C":
        # Simple Sharpen + ESRGAN
        sharp = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        output = esrgan_process(sharp)
    
    else:
        raise HTTPException(400, "Type must be A, B, or C")

    # Return Image
    buf = io.BytesIO()
    output.save(buf, format="JPEG")
    buf.seek(0)
    
    # Optional: Memory cleanup after heavy request
    gc.collect()

    return StreamingResponse(buf, media_type="image/jpeg")

@app.get("/")
def home():
    return {"status": "Online", "message": "Models will load on first request"}
