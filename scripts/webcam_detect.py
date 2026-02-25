import cv2
import numpy as np
import tensorflow as tf
import os

# Classes (Copied from app.py for standalone usage)
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

def start_webcam():
    model_path = 'models/road_sign_model.h5'
    if not os.path.exists(model_path):
        print("Model not found! Running with dummy recognition.")
        model = None
    else:
        print("Loading real model...")
        model = tf.keras.models.load_model(model_path)

    # Initialize Webcam (Try index 0, 1, or 2 for Mac compatibility)
    cap = None
    for i in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"Using camera index {i}")
            break
    
    if cap is None:
        print("Error: Could not open any webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process frame for prediction
        # We take the center crop or just resize
        h, w, _ = frame.shape
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        roi = frame[start_y:start_y+size, start_x:start_x+size]
        
        # Prepare for model
        input_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (30, 30))
        input_img = np.expand_dims(input_img, axis=0) # No /255 as per current model training
        
        # Prediction
        label = "Scanning..."
        confidence = 0
        
        if model:
            preds = model.predict(input_img, verbose=0)
            class_idx = np.argmax(preds[0])
            confidence = np.max(preds[0])
            if confidence > 0.8: # Threshold
                label = f"{CLASSES[class_idx]} ({confidence*100:.1f}%)"
            else:
                label = "Unknown Sign"

        # Display results on frame
        color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (start_x+size, start_y+size), color, 2)
        
        cv2.imshow('RoadSign AI - Realtime', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam()
