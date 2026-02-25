# PathOracle - The AI Visionary of the Highway

A Deep Learning suite using multi-model neural networks into classify traffic signs, parse scene segmentation, and perform live OCR analysis.

## Features
- **Modern Web Interface**: Upload and classify signs instantly.
- **High Performance CNN**: Multi-layered architecture for accurate recognition.
- **Dataset Integration**: Scripts provided for GTSRB (German Traffic Sign Recognition Benchmark).

## Project Structure
- `app.py`: Flask backend server.
- `scripts/download_data.py`: Downloads and prepares the GTSRB dataset.
- `scripts/train.py`: Defines the CNN architecture and handles training.
- `scripts/setup_dev.py`: Generates a dummy model for quick UI testing.
- `templates/`: HTML frontend files.
- `models/`: Location where the trained model (`road_sign_model.h5`) is stored.

## Quick Start (Development Mode)
This mode uses a dummy model to test the web interface.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate a dummy model:
   ```bash
   python3 scripts/setup_dev.py
   ```
3. Run the web app:
   ```bash
   python3 app.py
   ```
4. Open your browser at `http://localhost:5001`.

## Full Training Pipeline
To train a real model with the GTSRB dataset:

1. Download the data:
   ```bash
   python3 scripts/download_data.py
   ```
2. Start training:
   ```bash
   python3 scripts/train.py
   ```
3. Once training finishes, restart the server to use the new model.

## Technology Stack
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Backend**: Flask
- **Frontend**: Glassmorphism CSS, Vanilla JS
