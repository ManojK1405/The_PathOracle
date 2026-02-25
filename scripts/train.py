import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

def train_model(data_dir='data'):
    data = []
    labels = []
    classes = 43
    
    # Path to training images
    train_path = os.path.join(data_dir, 'Train')
    
    if not os.path.exists(train_path):
        print(f"Directory {train_path} not found. Please download the dataset first.")
        return

    print("Loading images...")
    for i in range(classes):
        path = os.path.join(train_path, str(i).zfill(5))
        if not os.path.exists(path):
            # Fallback for unpadded names
            path = os.path.join(train_path, str(i))
            
        images = os.listdir(path)
        
        for a in images:
            try:
                image = cv2.imread(os.path.join(path, a))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (30, 30))
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Error loading image {a}: {e}")
                
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"Found {len(data)} images across {classes} classes.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Data Augmentation
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False, # Traffic signs are directional
        vertical_flip=False,
        fill_mode="nearest"
    )
    
    # Build the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train with Augmentation
    print("Starting training with Data Augmentation...")
    epochs = 20 # Increased epochs since augmentation makes learning harder but better
    batch_size = 32
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_test, y_test),
        epochs=epochs
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save("models/road_sign_model.h5")
    print("Model saved to models/road_sign_model.h5")
    
    return history

if __name__ == "__main__":
    train_model()
