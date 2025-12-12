import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

def apply_pro_enhancement(image_bytes):
    # Image read karna
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None

    # --- STEP 1: INTELLIGENT HDR (CLAHE) ---
    # Hum image ko LAB color space me convert karenge (Ye human eye ke qareeb hota hai)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Sirf 'Lightness' channel par CLAHE lagayenge (Colors kharab nahi honge)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Wapis merge karo
    limg = cv2.merge((cl, a, b))
    hdr_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # --- STEP 2: SOFT SKIN GLOW & WHITENING ---
    # Thora sa Brightness barhana (Gora karne ke liye)
    # Lekin Soft light ke sath mix karenge taake details na jain
    hsv = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Saturation (Rang) thora sa tez (Lekin 40 nahi, sirf 15 taake over na lage)
    s = cv2.add(s, 15) 
    # Value (Brightness) thori si barhai
    v = cv2.add(v, 15)
    
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # --- STEP 3: PROFESSIONAL SHARPENING (Unsharp Masking) ---
    # Ye technique Photoshop use karta hai.
    # Hum original image ka blur version banayenge
    gaussian_blur = cv2.GaussianBlur(bright_img, (0, 0), 2.0)
    # Phir original aur blur ko mix karke sharp look nikalenge (Weighted Add)
    sharp_img = cv2.addWeighted(bright_img, 1.3, gaussian_blur, -0.3, 0)

    # --- FINAL TOUCH: NOISE REMOVAL ---
    # Agar thora bohot daana (grain) aa gya ho to halka sa safaya
    # Ye step thora slow ho sakta hai magar quality deta hai. 
    # Agar speed chahiye to ye line hata den, lekin quality ke liye rehne den.
    final_output = cv2.fastNlMeansDenoisingColored(sharp_img, None, 3, 3, 7, 21)

    # Image wapis encode karna
    _, encoded_img = cv2.imencode('.jpg', final_output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return io.BytesIO(encoded_img.tobytes())

@app.route('/api/enhance', methods=['GET', 'POST'])
def enhance_api():
    image_bytes = None
    
    # --- GET REQUEST ---
    if request.method == 'GET':
        image_url = request.args.get('url')
        if not image_url:
            return jsonify({"error": "URL missing"}), 400
        try:
            resp = requests.get(image_url)
            if resp.status_code == 200:
                image_bytes = resp.content
            else:
                return jsonify({"error": "Failed to fetch image"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # --- POST REQUEST ---
    elif request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        elif request.json and 'url' in request.json:
            image_url = request.json['url']
            resp = requests.get(image_url)
            image_bytes = resp.content
        elif 'url' in request.form:
             image_url = request.form['url']
             resp = requests.get(image_url)
             image_bytes = resp.content
        else:
            return jsonify({"error": "No data provided"}), 400

    # PROCESS
    if image_bytes:
        try:
            processed_image = apply_pro_enhancement(image_bytes)
            if processed_image:
                return send_file(processed_image, mimetype='image/jpeg')
            else:
                return jsonify({"error": "Processing failed"}), 400
        except Exception as e:
             return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No image found"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
    
