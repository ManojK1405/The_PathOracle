import os
import cv2
import random

def convert_samples():
    src_dir = 'data/Train'
    dest_dir = 'test_samples'
    os.makedirs(dest_dir, exist_ok=True)
    
    classes = os.listdir(src_dir)
    # Pick 10 random classes
    sample_classes = random.sample([c for c in classes if os.path.isdir(os.path.join(src_dir, c))], 10)
    
    for c in sample_classes:
        class_path = os.path.join(src_dir, c)
        images = [f for f in os.listdir(class_path) if f.endswith('.ppm')]
        if images:
            img_name = random.choice(images)
            img = cv2.imread(os.path.join(class_path, img_name))
            new_name = f"class_{int(c)}_{img_name.replace('.ppm', '.png')}"
            cv2.imwrite(os.path.join(dest_dir, new_name), img)
            print(f"Converted {new_name}")

if __name__ == "__main__":
    convert_samples()
