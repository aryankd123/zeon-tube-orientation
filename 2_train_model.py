"""
Model Training Script.
Utilizes the pre-trained YOLOv8 Nano Pose model to detect tube joints and orientation tabs.
"""

from ultralytics import YOLO
import os

def train_model():
    print("Initializing YOLOv8n-pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    yaml_path = os.path.abspath(os.path.join("split_dataset_v2", "data.yaml"))
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("data.yaml not found. Run 1_data_preparation.py first.")

    print("Starting training process...")
    # Training parameters optimized for a small dataset and fast convergence
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        device='auto', # Automatically select GPU if available
        project="runs",
        name="tube_detection_model"
    )
    
    print("Training complete. Weights saved in the 'runs' directory.")

if __name__ == "__main__":
    train_model()