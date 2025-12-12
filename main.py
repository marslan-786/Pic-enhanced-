from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests
import io
from PIL import Image, ImageFilter
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from realesrgan import RealESRGAN
from gfpgan import GFPGANer
import torch

app = FastAPI()


# --------------------------------------------
# MODE A: GFPGAN + ESRGAN
# --------------------------------------------
print("Loading Mode A models...")

gfpgan = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    upscale=2
)

def esrgan_upscale(img):
    model = RealESRGAN(torch.device("cpu"), scale=4)
    model.load_weights('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x4plus.pth')
    return model.predict(img)


# --------------------------------------------
# MODE B: Stable Diffusion Img2Img
# --------------------------------------------
print("Loading Mode B SD pipeline...")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")


# --------------------------------------------
# MODE C: SHARPEN + DENOISE + ESRGAN (WITHOUT CV2)
# --------------------------------------------
def mode_c_process(image: Image.Image):
    
    # Sharpen
    sharp = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Light denoise (Pillow)
    denoise = sharp.filter(ImageFilter.SMOOTH_MORE)

    # Convert to ESRGAN
    upscaled = esrgan_upscale(denoise)

    return upscaled


# --------------------------------------------
# Download Helper
# --------------------------------------------
def download_image(url):
    try:
        content = requests.get(url, timeout=10).content
        return Image.open(io.BytesIO(content)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image URL")


# --------------------------------------------
# MAIN API
# --------------------------------------------
@app.get("/enhance")
def enhance(type: str, url: str):

    image = download_image(url)

    type = type.upper()

    if type == "A":
        print("Running Mode A...")
        _, _, restored = gfpgan.enhance(np.array(image), has_aligned=False)
        restored = Image.fromarray(restored)
        output = esrgan_upscale(restored)

    elif type == "B":
        print("Running Mode B...")
        output = pipe(
            prompt="ultra detailed anime style, clear face, HQ",
            image=image,
            strength=0.65,
            guidance_scale=7.0
        ).images[0]

    elif type == "C":
        print("Running Mode C...")
        output = mode_c_process(image)

    else:
        raise HTTPException(400, "Type must be A, B, or C")

    # Return image
    buf = io.BytesIO()
    output.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/")
def home():
    return {"status": "Enhancer Active: A / B / C"}
