# app.py
import os
import io
import sys
import time
import math
import torch
import logging
import tempfile
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from flask import Flask, request, jsonify, send_file

# Optional high-perf libs (used if available)
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except Exception:
    HAS_GFPGAN = False

try:
    from realesrgan import RealESRGAN
    HAS_REALESRGAN = True
except Exception:
    HAS_REALESRGAN = False

import cv2

# ---------------------------
# Basic configuration
# ---------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on device: {DEVICE}")

# Model directories (will auto-create)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# Initialize GFPGAN (face restore) if available
# ---------------------------
gfpganer = None
if HAS_GFPGAN:
    try:
        # Default model name; GFPGAN will auto-download if not present (depending on version)
        gfpganer = GFPGANer(
            model_path=os.path.join(MODEL_DIR, "GFPGANv1.4.pth"),
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=DEVICE
        )
        logging.info("GFPGAN initialized.")
    except Exception as e:
        logging.warning(f"GFPGAN init failed: {e}")
        gfpganer = None

# ---------------------------
# Initialize Real-ESRGAN (upscaler) if available
# ---------------------------
realesrganer = None
if HAS_REALESRGAN:
    try:
        # Choose model; "RealESRGAN_x4plus" common
        realesrganer = RealESRGAN(device=DEVICE, scale=4)
        model_path = os.path.join(MODEL_DIR, "RealESRGAN_x4plus.pth")
        realesrganer.load_weights(model_path)
        logging.info("Real-ESRGAN initialized.")
    except Exception as e:
        logging.warning(f"Real-ESRGAN init failed: {e}")
        realesrganer = None

# ---------------------------
# Helper utilities
# ---------------------------
def read_image_from_bytes(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_to_bytes(cv2_img, quality=95):
    _, enc = cv2.imencode('.jpg', cv2_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return io.BytesIO(enc.tobytes())

def pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert('RGB'))[:, :, ::-1]
    return arr

def cv2_to_pil(cv2_img):
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# A stronger, but controllable enhancement pipeline
def pro_enhance_pipeline(cv2_img, target_upscale=1, face_restore=True, enhance_strength=0.9):
    """
    cv2_img: BGR numpy array
    target_upscale: integer (1 means keep, 2 or 4 uses Real-ESRGAN if present)
    face_restore: use GFPGAN if available
    enhance_strength: 0..1 how aggressive beautify should be
    """
    # Step 0: keep copy
    src = cv2_img.copy()

    # Step 1: Gentle denoise (to reduce sensor artifacts but not blur)
    # Non-local means with moderate parameters
    denoised = cv2.fastNlMeansDenoisingColored(src, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)

    # Step 2: LAB CLAHE for controlled HDR pop
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.9 + 0.6 * enhance_strength, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_merged = cv2.merge((cl, a, b))
    hdr = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    # Step 3: Vibrancy and brightness (HSV), keep safe clipping
    hsv = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv)
    s = np.clip(s + int(18 * enhance_strength), 0, 255).astype(np.uint8)
    v = np.clip(v + int(22 * enhance_strength), 0, 255).astype(np.uint8)
    hsv2 = cv2.merge((h.astype(np.uint8), s, v))
    vibrant = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    # Step 4: Whitening / skin tone shift (selective)
    # We'll do a simple skin detection on HSV and increase brightness + reduce saturation slightly to get white-skin look
    hsv3 = cv2.cvtColor(vibrant, cv2.COLOR_BGR2HSV)
    h3, s3, v3 = cv2.split(hsv3)

    # Skin color range (rough) - tuned to include varied skin tones
    lower = np.array([0, 10, 60])
    upper = np.array([25, 200, 255])
    skin_mask = cv2.inRange(hsv3, lower, upper)
    mask_blur = cv2.GaussianBlur(skin_mask, (7,7), 0)

    # Create whitening effect by boosting V and slightly desaturating in skin areas
    v3 = v3.astype(np.int16)
    s3 = s3.astype(np.int16)
    # Whitening multiplier
    v3 = np.clip(v3 + (18 * enhance_strength) * (mask_blur / 255.0), 0, 255).astype(np.uint8)
    s3 = np.clip(s3 - (12 * enhance_strength) * (mask_blur / 255.0), 0, 255).astype(np.uint8)
    hsv_skin = cv2.merge([h3, s3, v3])
    whitened = cv2.cvtColor(hsv_skin, cv2.COLOR_HSV2BGR)

    # Step 5: Local contrast enhancement (unsharp mask but safe)
    # Unsharp mask: original + amount*(original - blur(original))
    blur = cv2.GaussianBlur(whitened, (0,0), 1.2 + 0.6*(1-enhance_strength))
    amount = 1.25 + 0.6*enhance_strength
    sharp = cv2.addWeighted(whitened, 1.0 + amount, blur, -amount, 0)

    # Step 6: Micro-contrast (CLAHE on luminance channel small)
    lab2 = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l2, a2, b2 = cv2.split(lab2)
    clahe2 = cv2.createCLAHE(clipLimit=1.2 + 0.6*enhance_strength, tileGridSize=(6,6))
    cl2 = clahe2.apply(l2)
    lab2 = cv2.merge((cl2, a2, b2))
    micro = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Step 7: Very light bilateral to preserve edges while smoothing skin slightly
    beauty = cv2.bilateralFilter(micro, d=5, sigmaColor=50 * (1 - 0.2*enhance_strength), sigmaSpace=50)

    result = beauty

    # Step 8: Face restore (GFPGAN) — use PIL bridge
    if face_restore and HAS_GFPGAN and gfpganer is not None:
        try:
            pil_img = cv2_to_pil(result)
            # GFPGAN expects RGB
            cropped_faces, restored_img = gfpganer.enhance(np.array(pil_img)[:, :, ::-1], has_aligned=False, only_center_face=False, paste_back=True)
            # restored_img is BGR
            result = restored_img
        except Exception as e:
            logging.warning(f"GFPGAN restore failed: {e}")

    # Step 9: Upscale via Real-ESRGAN if requested
    if target_upscale >= 2 and HAS_REALESRGAN and realesrganer is not None:
        try:
            # Real-ESRGAN expects PIL or numpy, using PIL
            pil_for_up = cv2_to_pil(result)
            sr_img = realesrganer.predict(pil_for_up)
            # realesrgan returns PIL
            result = pil_to_cv2(sr_img)
        except Exception as e:
            logging.warning(f"Real-ESRGAN upscaling failed: {e}")

    # Step 10: Final color grade - warm highlight + slight vignette removal (subtle)
    final = result.copy()
    # small global contrast and saturation boost using PIL
    pil_final = cv2_to_pil(final)
    enhancer = ImageEnhance.Color(pil_final)
    pil_final = enhancer.enhance(1.08 + 0.2*enhance_strength)
    enhancer = ImageEnhance.Brightness(pil_final)
    pil_final = enhancer.enhance(1.04 + 0.1*enhance_strength)
    final = pil_to_cv2(pil_final)

    return final

# ---------------------------
# API Endpoint
# ---------------------------
@app.route("/api/enhance", methods=["GET", "POST"])
def enhance_api():
    """
    GET: /api/enhance?url=<image_url>&upscale=2&face=1&strength=0.9
    POST: form-data file=@image.jpg OR json {"url":"..."}
    """
    try:
        image_bytes = None

        # GET url
        if request.method == "GET":
            image_url = request.args.get("url")
            if not image_url:
                return jsonify({"error": "URL missing"}), 400
            import requests
            headers = {"User-Agent":"Mozilla/5.0"}
            resp = requests.get(image_url, headers=headers, timeout=20)
            if resp.status_code != 200:
                return jsonify({"error": f"Could not fetch image, status {resp.status_code}"}), 400
            image_bytes = resp.content

        # POST: file or json url
        elif request.method == "POST":
            if "file" in request.files:
                image_bytes = request.files["file"].read()
            elif request.is_json and request.json.get("url"):
                import requests
                resp = requests.get(request.json.get("url"), timeout=20, headers={"User-Agent":"Mozilla/5.0"})
                if resp.status_code != 200:
                    return jsonify({"error": "Failed to fetch URL"}), 400
                image_bytes = resp.content
            elif request.form.get("url"):
                import requests
                resp = requests.get(request.form.get("url"), timeout=20, headers={"User-Agent":"Mozilla/5.0"})
                if resp.status_code != 200:
                    return jsonify({"error": "Failed to fetch URL"}), 400
                image_bytes = resp.content
            else:
                return jsonify({"error":"No image provided"}), 400

        # Params
        upscale = int(request.args.get("upscale", request.form.get("upscale", 1)))
        face = int(request.args.get("face", request.form.get("face", 1)))
        strength = float(request.args.get("strength", request.form.get("strength", 0.92)))

        # Safety clamps
        upscale = max(1, min(4, upscale))
        strength = max(0.0, min(1.0, strength))
        face = 1 if face else 0

        # Read image
        cv2_img = read_image_from_bytes(image_bytes)
        if cv2_img is None:
            return jsonify({"error":"Invalid image data"}), 400

        # Process
        start = time.time()
        out_cv2 = pro_enhance_pipeline(cv2_img, target_upscale=upscale, face_restore=bool(face), enhance_strength=strength)
        elapsed = time.time() - start
        logging.info(f"Processed in {elapsed:.2f}s")

        # Return
        return send_file(cv2_to_bytes(out_cv2, quality=95), mimetype="image/jpeg")
    except Exception as e:
        logging.exception("Processing error")
        return jsonify({"error": str(e)}), 500

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
