import os
import requests
import zipfile
import shutil

def download_file(url, filename):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
    print(f"Finished downloading {filename}")

def setup_dataset():
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for training images
    # Note: Using a slightly smaller or alternative link if possible, 
    # but the official one is the most reliable.
    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    zip_path = os.path.join(data_dir, "GTSRB_Final_Training_Images.zip")
    
    if not os.path.exists(zip_path):
        download_file(train_url, zip_path)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    # The zip contains a folder 'GTSRB/Final_Training/Images' - we want it to be flatter
    # so scripts/train.py can find it easily.
    src_images = os.path.join(data_dir, 'GTSRB', 'Final_Training', 'Images')
    dest_images = os.path.join(data_dir, 'Train')
    
    if os.path.exists(src_images):
        if os.path.exists(dest_images):
            shutil.rmtree(dest_images)
        shutil.move(src_images, dest_images)
        print(f"Moved images to {dest_images}")
        
    print("Dataset setup complete.")

if __name__ == "__main__":
    setup_dataset()
