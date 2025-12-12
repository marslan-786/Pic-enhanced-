from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests
import io
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from realesrgan import RealESRGAN
from gfpgan import GFPGANer
import torch

app = FastAPI()

# --------------------------------------------
# MODE A: GFPGAN + Real ESRGAN (HDR + Face Enhance)
# --------------------------------------------
print("Loading Mode A models...")
gfpgan = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    upscale=2
)

# Real ESRGAN for Super Resolution
def esrgan_upscale(img):
    model = RealESRGAN(torch.device("cpu"), scale=4)
    model.load_weights('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x4plus.pth')
    return model.predict(img)

# --------------------------------------------
# MODE B: Stable Diffusion Img2Img (Cartoon / Anime)
# --------------------------------------------
print("Loading Mode B SD pipeline...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

# --------------------------------------------
# MODE C: OpenCV Sharpen + Noise Reduce + ESRGAN
# --------------------------------------------
def mode_c_process(image):
    img = np.array(image)

    # Sharpen Filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)

    # Denoise
    denoise = cv2.fastNlMeansDenoisingColored(sharp, None, 10, 10, 7, 21)

    # Upscale using ESRGAN
    upscaled = esrgan_upscale(Image.fromarray(denoise))

    return upscaled


# --------------------------------------------
# Download Image Helper
# --------------------------------------------
def download_image(url):
    try:
        raw = requests.get(url, timeout=10).content
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid Image URL")


# --------------------------------------------
# API Endpoint
# --------------------------------------------
@app.get("/enhance")
def enhance(type: str, url: str):
    image = download_image(url)

    # MODE SELECTION
    if type.upper() == "A":
        print("Running Mode A...")
        _, _, output = gfpgan.enhance(np.array(image), has_aligned=False)
        output = Image.fromarray(output)
        output = esrgan_upscale(output)

    elif type.upper() == "B":
        print("Running Mode B...")
        output = pipe(
            prompt="high quality anime style face, ultra detailed, clear skin",
            image=image,
            strength=0.65,
            guidance_scale=7.5
        ).images[0]

    elif type.upper() == "C":
        print("Running Mode C...")
        output = mode_c_process(image)

    else:
        raise HTTPException(400, "Invalid type (use A, B, or C)")

    # Return image
    buf = io.BytesIO()
    output.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/")
def home():
    return {"status": "Three Mode Enhancer Active (A,B,C)"}
