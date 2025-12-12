import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

def apply_ultimate_enhancement(image_bytes):
    # --- STAGE 1: Reading Image ---
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    # --- STAGE 2: ADVANCED HDR (Stronger CLAHE) ---
    # LAB colorspace for better luminance control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit increased to 3.0 for more dramatic HDR effect
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    hdr_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # --- STAGE 3: VIBRANT BRIGHTNESS & WHITENING ---
    hsv = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increased Saturation (+20) for deeper colors (Kapray/Background)
    s = cv2.add(s, 20) 
    # Increased Value (+30) for brighter, whiter skin look
    v = cv2.add(v, 30)
    
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # --- STAGE 4: PRO SHARPENING (Unsharp Mask) ---
    # Keeps details crisp without destroying them
    gaussian_blur = cv2.GaussianBlur(bright_img, (0, 0), 2.0)
    sharp_img = cv2.addWeighted(bright_img, 1.4, gaussian_blur, -0.4, 0)

    # --- STAGE 5: Final Smoothing & Noise Removal ---
    # Strength (h) increased slightly to 5 for smoother "porcelain" skin
    final_output = cv2.fastNlMeansDenoisingColored(sharp_img, None, 5, 5, 7, 21)

    # Encoding with high JPEG quality (98%)
    _, encoded_img = cv2.imencode('.jpg', final_output, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return io.BytesIO(encoded_img.tobytes())

# ================= API ROUTES (Same as before) =================

@app.route('/api/enhance', methods=['GET', 'POST'])
def enhance_api():
    image_bytes = None
    
    # --- GET REQUEST (URL Link) ---
    if request.method == 'GET':
        image_url = request.args.get('url')
        if not image_url: return jsonify({"error": "URL missing"}), 400
        try:
            resp = requests.get(image_url, timeout=10)
            if resp.status_code == 200: image_bytes = resp.content
            else: return jsonify({"error": f"Failed to fetch image, status: {resp.status_code}"}), 400
        except Exception as e: return jsonify({"error": str(e)}), 400

    # --- POST REQUEST (File or URL) ---
    elif request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        elif request.json and 'url' in request.json:
            image_url = request.json['url']
            resp = requests.get(image_url, timeout=10)
            image_bytes = resp.content
        elif 'url' in request.form:
             image_url = request.form['url']
             resp = requests.get(image_url, timeout=10)
             image_bytes = resp.content
        else:
            return jsonify({"error": "No image data provided"}), 400

    # --- PROCESSING ---
    if image_bytes:
        try:
            # Using the new ULTIMATE function
            processed_image = apply_ultimate_enhancement(image_bytes)
            if processed_image:
                return send_file(processed_image, mimetype='image/jpeg')
            else:
                return jsonify({"error": "Processing failed, image might be corrupt"}), 400
        except Exception as e:
             return jsonify({"error": f"Server Error: {str(e)}"}), 500
    else:
        return jsonify({"error": "Could not get image bytes"}), 400

if __name__ == '__main__':
    # Railway Port Configuration
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
