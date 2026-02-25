import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Optional: suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
import json
from io import BytesIO
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Road Sign Classes (GTSRB)
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next junction', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road works',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing for vehicles over 3.5 metric tons'
}

from deep_translator import GoogleTranslator

# Global variable for models
model = None
MODEL_PATH = 'models/road_sign_model.h5'
ADV_MODELS = {'yolo': None, 'ocr': None, 'seg': None}

def get_model():
    global model
    global model_error
    if 'model_error' not in globals():
        model_error = None
        
    if model is None:
        if os.path.exists(MODEL_PATH) and tf:
            try:
                # Handle Keras version differences (batch_shape vs batch_input_shape)
                class CustomInputLayer(tf.keras.layers.InputLayer):
                    def __init__(self, **kwargs):
                        if 'batch_shape' in kwargs:
                            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                        super().__init__(**kwargs)
                
                model = load_model(MODEL_PATH, custom_objects={'InputLayer': CustomInputLayer})
                print("Model loaded successfully.")
                model_error = None
            except Exception as e:
                model_error = str(e)
                print(f"Error loading model: {e}")
        else:
            model_error = "Model file not found or TensorFlow not imported."
            print(model_error)
    return model

def get_yolo():
    if ADV_MODELS['yolo'] is None:
        from ultralytics import YOLO
        ADV_MODELS['yolo'] = YOLO('yolov8n.pt')
    return ADV_MODELS['yolo']

def get_ocr():
    if ADV_MODELS['ocr'] is None:
        import easyocr
        ADV_MODELS['ocr'] = easyocr.Reader(['en'])
    return ADV_MODELS['ocr']

def get_seg():
    if ADV_MODELS['seg'] is None:
        import torch
        from torchvision import models
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT').eval()
        ADV_MODELS['seg'] = model
    return ADV_MODELS['seg']

def advanced_process(img_bgr, features):
    results = {}
    
    # Feature 4: Night Enhancement (Night-Vision)
    if features.get('nightVision'):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        _, buffer = cv2.imencode('.jpg', enhanced)
        results['enhanced_img'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        img_bgr = enhanced # Use enhanced for rest

    # Feature 1: Multi-Sign Detection
    if features.get('multiDetect'):
        try:
            yolo = get_yolo()
            yolo_res = yolo(img_bgr, verbose=False)[0]
            results['detections'] = len(yolo_res.boxes)
        except: results['detections'] = 0

    # Feature 3: Plate OCR
    if features.get('ocr'):
        try:
            ocr = get_ocr()
            ocr_res = ocr.readtext(img_bgr)
            results['ocr'] = [r[1] for r in ocr_res if r[2] > 0.3]
        except: results['ocr'] = []

    # Feature 2: Sign Health Assessment
    if features.get('health'):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        avg_sat = np.mean(hsv[:,:,1])
        sharp_score = min(100, laplacian_var / 5)
        color_score = min(100, avg_sat * 2)
        total_score = int((sharp_score + color_score) / 2)
        results['health'] = {
            'score': total_score,
            'status': 'Healthy' if total_score > 60 else 'Maintenance Required'
        }

    # Feature 6: Semantic Segmentation
    if features.get('segmentation'):
        try:
            import torch
            from torchvision import transforms
            seg_model = get_seg()
            input_tensor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(img_bgr).unsqueeze(0)
            with torch.no_grad():
                output = seg_model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()
            mask = np.zeros((256, 256, 3), dtype=np.uint8)
            mask[output_predictions == 15] = [255, 0, 0] # Person/Sign approx in mobile
            img_small = cv2.resize(img_bgr, (256, 256))
            blended = cv2.addWeighted(img_small, 0.5, mask, 0.5, 0)
            _, buffer = cv2.imencode('.jpg', blended)
            results['segmented_img'] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        except: pass

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    lang = request.form.get('lang', 'en')
    features_raw = request.form.get('features', '{}')
    features = json.loads(features_raw)
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_bgr = cv2.imread(filepath)
        
        # Advanced Processing
        advanced_results = advanced_process(img_bgr, features)
        
        # Original Prediction Logic (on 30x30 crop)
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (30, 30))
            img_tensor = np.expand_dims(img_resized, axis=0)
            
            m = get_model()
            if m:
                preds = m.predict(img_tensor)
                top_3_indices = np.argsort(preds[0])[-3:][::-1]
                predictions = []
                for idx in top_3_indices:
                    prob = float(preds[0][idx])
                    original_label = CLASSES.get(idx, "Unknown")
                    try:
                        translated_label = GoogleTranslator(source='auto', target=lang).translate(original_label) if lang != 'en' else original_label
                    except: translated_label = original_label
                    predictions.append({'label': translated_label, 'original': original_label, 'confidence': prob})
                
                return jsonify({
                    'success': True,
                    'top_predictions': predictions,
                    'prediction': predictions[0]['label'],
                    'confidence': predictions[0]['confidence'],
                    'original': predictions[0]['original'],
                    'advanced': advanced_results
                })
            else:
                # Mock result (maintained for local logic testing)
                mock_indices = np.random.choice(range(43), 3, replace=False)
                predictions = []
                for i, idx in enumerate(mock_indices):
                    mock_text = CLASSES[idx]
                    try: t_text = GoogleTranslator(source='auto', target=lang).translate(mock_text) if lang != 'en' else mock_text
                    except: t_text = mock_text
                    predictions.append({'label': f"[MOCK] {t_text}", 'original': CLASSES[idx], 'confidence': 0.95 - (i * 0.2)})
                return jsonify({
                    'success': True,
                    'top_predictions': predictions,
                    'prediction': predictions[0]['label'],
                    'confidence': predictions[0]['confidence'],
                    'original': predictions[0]['original'],
                    'advanced': advanced_results,
                    'model_type': 'Oracle Simulator',
                    'error': globals().get('model_error', 'Unknown error')
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
