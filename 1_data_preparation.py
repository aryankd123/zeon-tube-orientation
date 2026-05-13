"""
Data Preparation Script for Zeon Systems Assignment.
This script extracts the dataset, performs an 80-20 train-validation split,
and formats the CSV annotations into YOLOv8-Pose (.txt) format.
"""

import os
import shutil
import random
import glob
import zipfile
import pandas as pd
import yaml

def prepare_data():
    print("Starting data preparation pipeline...")

    # 1. Extract ZIP file if not already extracted
    zip_path = "dataset/zeon_yolo_dataset.zip"
    extract_dir = "dataset/extracted"
    if os.path.exists(zip_path) and not os.path.exists(extract_dir):
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    # 2. Locate Images and Annotations automatically
    csv_files = glob.glob(f"{extract_dir}/**/annotations.csv", recursive=True)
    if not csv_files:
        raise FileNotFoundError("annotations.csv not found in the extracted dataset.")
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    first_img_name = df['image'].iloc[0]
    
    img_files = glob.glob(f"{extract_dir}/**/{first_img_name}", recursive=True)
    orig_images_dir = os.path.dirname(img_files[0])
    orig_labels_dir = os.path.dirname(glob.glob(f"{extract_dir}/**/{first_img_name.replace('.png', '.txt')}", recursive=True)[0])

    # 3. Create Split Directories (Train/Val)
    split_base = "split_dataset_v2"
    dirs = [
        os.path.join(split_base, "images", "train"),
        os.path.join(split_base, "images", "val"),
        os.path.join(split_base, "labels", "train"),
        os.path.join(split_base, "labels", "val")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 4. Perform 80-20 Split
    all_images = [f for f in os.listdir(orig_images_dir) if f.endswith(('.png', '.jpg'))]
    random.seed(42) # Ensure reproducibility
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)
    train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]

    def transfer_files(file_list, target_img_dir, target_lbl_dir):
        for img in file_list:
            shutil.copy(os.path.join(orig_images_dir, img), os.path.join(target_img_dir, img))
            txt_file = os.path.splitext(img)[0] + ".txt"
            if os.path.exists(os.path.join(orig_labels_dir, txt_file)):
                shutil.copy(os.path.join(orig_labels_dir, txt_file), os.path.join(target_lbl_dir, txt_file))

    transfer_files(train_imgs, dirs[0], dirs[2])
    transfer_files(val_imgs, dirs[1], dirs[3])

    # 5. Generate data.yaml for YOLOv8
    yaml_path = os.path.join(split_base, "data.yaml")
    yaml_config = {
        'path': os.path.abspath(split_base),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'tube'},
        'kpt_shape': [2, 3] # 2 Keypoints (Center, Tab), 3 Dimensions (x, y, visibility)
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print(f"Data preparation complete! Train: {len(train_imgs)} | Val: {len(val_imgs)}")

if __name__ == "__main__":
    prepare_data()