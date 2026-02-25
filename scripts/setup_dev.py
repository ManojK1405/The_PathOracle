import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

def create_dummy_model():
    print("Creating dummy model for development...")
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30, 30, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(43, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    os.makedirs('models', exist_ok=True)
    model.save("models/road_sign_model.h5")
    print("Dummy model saved to models/road_sign_model.h5")

if __name__ == "__main__":
    create_dummy_model()
