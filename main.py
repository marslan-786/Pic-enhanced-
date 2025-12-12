from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import requests
import io
from PIL import Image
import numpy as np
import cv2
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import gc

app = FastAPI()

# Global Models
face_enhancer = None
bg_enhancer = None

def load_models():
    global face_enhancer, bg_enhancer
    
    if face_enhancer is None:
        print("💎 Loading Models (This creates the Magic)...")
        
        # 1. Load Background Enhancer (RealESRGAN)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_enhancer = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False # CPU mode
        )

        # 2. Load Face Enhancer (GFPGAN - The Remini Logic)
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_enhancer # Link background enhancer here
        )
        print("✅ Models Loaded Successfully!")

    return face_enhancer

@app.get("/enhance")
def enhance(url: str):
    print(f"⚡ Processing: {url}")
    
    try:
        # 1. Download Image
        resp = requests.get(url, stream=True, timeout=20)
        resp.raise_for_status()
        
        # 2. Convert to CV2 format
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Could not decode image"}

        # 3. Get Model
        restorer = load_models()

        # 4. RUN ENHANCEMENT (The Remini Step)
        # has_aligned=False means it will detect faces automatically
        # only_center_face=False means it will fix ALL faces
        # paste_back=True means it puts the fixed face back on the body
        _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        # 5. Convert back to JPEG
        _, encoded_img = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # 6. Memory Cleanup
        gc.collect()

        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

@app.get("/")
def home():
    return {"status": "Remini-Style Enhancer Online"}
