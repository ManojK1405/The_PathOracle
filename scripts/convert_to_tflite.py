import tensorflow as tf
import os

def convert_to_tflite(h5_model_path='models/road_sign_model.h5', output_path='models/road_sign_model.tflite'):
    print(f"Loading Keras model from {h5_model_path}...")
    if not os.path.exists(h5_model_path):
        print("Error: .h5 model not found. Please train the model first.")
        return

    # Load the model
    model = tf.keras.models.load_model(h5_model_path)

    # Convert the model
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Success! TFLite model saved at {output_path}")

if __name__ == "__main__":
    convert_to_tflite()
