import cv2
import os  
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

def apply_ultra_enhancement(image_bytes):
    # Image ko read karna (Decode)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    # --- STEP 1: SKIN SMOOTHING ---
    smooth_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # --- STEP 2: WHITENING & BRIGHTNESS ---
    hsv = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 30) # Brightness barhai (Gora pan)
    
    # --- STEP 3: COLOR POP (SHOKH RANG) ---
    s = cv2.add(s, 40) # Saturation barhai (Colors pop)
    
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # --- STEP 4: ULTRA HDR EFFECT ---
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    hdr_img = cv2.filter2D(bright_img, -1, kernel)

    # Contrast & Brightness final touch
    alpha = 1.2
    beta = 10
    final_output = cv2.convertScaleAbs(hdr_img, alpha=alpha, beta=beta)

    # Image wapis encode karna response ke liye
    _, encoded_img = cv2.imencode('.jpg', final_output)
    return io.BytesIO(encoded_img.tobytes())

@app.route('/api/enhance', methods=['GET', 'POST'])
def enhance_api():
    image_bytes = None

    # 1. AGAR GET REQUEST HO (URL wala tareeqa)
    if request.method == 'GET':
        image_url = request.args.get('url') # URL query parameter se uthaye ga
        if not image_url:
            return jsonify({"error": "Bhai URL to dein (?url=...)"}), 400
        
        try:
            resp = requests.get(image_url)
            if resp.status_code == 200:
                image_bytes = resp.content
            else:
                return jsonify({"error": "URL se picture download nahi hui"}), 400
        except Exception as e:
            return jsonify({"error": f"Error downloading image: {str(e)}"}), 400

    # 2. AGAR POST REQUEST HO (Direct Picture Upload ya URL)
    elif request.method == 'POST':
        # Pehle check karein agar direct file upload ki hai
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        
        # Agar file nahi, to shayad JSON me URL bheja ho
        elif request.json and 'url' in request.json:
            image_url = request.json['url']
            resp = requests.get(image_url)
            image_bytes = resp.content
        
        # Agar form data me URL ho
        elif 'url' in request.form:
             image_url = request.form['url']
             resp = requests.get(image_url)
             image_bytes = resp.content
             
        else:
            return jsonify({"error": "Koi picture ya URL nahi mila POST request me"}), 400

    # --- FINAL PROCESSING ---
    if image_bytes:
        try:
            processed_image = apply_ultra_enhancement(image_bytes)
            if processed_image:
                # Direct Picture wapis bhej rahe hain (Jaisa aap ne kaha)
                return send_file(processed_image, mimetype='image/jpeg')
            else:
                return jsonify({"error": "Picture corrupt lag rahi hai, process nahi hui"}), 400
        except Exception as e:
             return jsonify({"error": f"Processing error: {str(e)}"}), 500
    else:
        return jsonify({"error": "Image data nahi mila"}), 400

# <--- Ye line sab se oopar imports ke sath likh den

# ... baqi sara code waisa hi rahe ga ...

if __name__ == '__main__':
    # Railway ka port uthao, agar na mile to 5000 use karo
    port = int(os.environ.get("PORT", 5000))
    # debug=False kar dein production ke liye
    app.run(debug=False, host='0.0.0.0', port=port)

    
