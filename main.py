import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

def apply_ultimate_pro_enhancement(image_bytes):
    # --- STAGE 1: Reading Image ---
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    # --- STAGE 2: STRONG PROFESSIONAL HDR (The Dramatic Look) ---
    # Using LAB color space to adjust lightness dramatically without messing up colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit set high (3.5) for a very strong, dramatic HDR contrast effect
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    hdr_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # --- STAGE 3: MAXIMUM VIBRANCY & BRIGHTNESS (The "Pop" & "Glow") ---
    hsv = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Significantly boost Saturation (+25) for deep, rich colors (orange top, ball)
    s = cv2.add(s, 25) 
    # significantly boost Value/Brightness (+35) for bright, glowing white skin look
    v = cv2.add(v, 35)
    
    final_hsv = cv2.merge((h, s, v))
    vibrant_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # --- STAGE 4: AGGRESSIVE PRO SHARPENING (The "Crispy" Details) ---
    # Using a strong Unsharp Mask technique to make details pop out
    gaussian_blur = cv2.GaussianBlur(vibrant_img, (0, 0), 3.0)
    # Higher weight (1.5) on the original image vs blur creates stronger sharpness
    sharp_img = cv2.addWeighted(vibrant_img, 1.5, gaussian_blur, -0.5, 0)

    # --- STAGE 5: FINAL POLISH (Porcelain Skin Finish) ---
    # Apply moderate denoising to ensure the skin looks smooth like porcelain, 
    # countering any grain introduced by the strong sharpening.
    final_output = cv2.fastNlMeansDenoisingColored(sharp_img, None, 6, 6, 7, 21)

    # Encoding with maximum JPEG quality
    _, encoded_img = cv2.imencode('.jpg', final_output, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return io.BytesIO(encoded_img.tobytes())

# ================= API ROUTES (Standard GET/POST) =================

@app.route('/api/enhance', methods=['GET', 'POST'])
def enhance_api():
    image_bytes = None
    
    # --- GET REQUEST (URL Link) ---
    if request.method == 'GET':
        image_url = request.args.get('url')
        if not image_url: return jsonify({"error": "URL missing"}), 400
        try:
            # Added User-Agent to avoid being blocked by some image hosts
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(image_url, headers=headers, timeout=15)
            if resp.status_code == 200: image_bytes = resp.content
            else: return jsonify({"error": f"Failed to fetch image, status: {resp.status_code}"}), 400
        except Exception as e: return jsonify({"error": f"Download error: {str(e)}"}), 400

    # --- POST REQUEST (File or URL) ---
    elif request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        elif request.json and 'url' in request.json:
            image_url = request.json['url']
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(image_url, headers=headers, timeout=15)
            image_bytes = resp.content
        elif 'url' in request.form:
             image_url = request.form['url']
             headers = {'User-Agent': 'Mozilla/5.0'}
             resp = requests.get(image_url, headers=headers, timeout=15)
             image_bytes = resp.content
        else:
            return jsonify({"error": "No image data provided in POST"}), 400

    # --- PROCESSING ---
    if image_bytes:
        try:
            # Using the new ULTIMATE PRO function
            processed_image = apply_ultimate_pro_enhancement(image_bytes)
            if processed_image:
                return send_file(processed_image, mimetype='image/jpeg')
            else:
                return jsonify({"error": "Processing failed, image might be corrupt/unsupported"}), 400
        except Exception as e:
             return jsonify({"error": f"Server processing error: {str(e)}"}), 500
    else:
        return jsonify({"error": "Could not get image bytes"}), 400

if __name__ == '__main__':
    # Railway Port Configuration
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
