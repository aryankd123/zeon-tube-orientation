"""
Model Evaluation and Inference Script.
Calculates Precision, Recall, F1 Score, and the Mean Absolute Error (MAE) of the rotation angle
using trigonometric functions (arctan2) on the predicted keypoints.
"""

import os
import glob
import pandas as pd
import numpy as np
from ultralytics import YOLO

def evaluate():
    print("Loading trained model...")
    # Update path if your run name changed
    model_path = os.path.join("runs", "tube_detection_model", "weights", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model weights not found. Train the model first.")
    
    model = YOLO(model_path)
    
    # Locate validation images and ground truth CSV
    val_dir = os.path.join("split_dataset_v2", "images", "val")
    csv_files = glob.glob("dataset/**/annotations.csv", recursive=True)
    df = pd.read_csv(csv_files[0])
    
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.png')]
    
    tp, fp, fn = 0, 0, 0
    angle_errors = []

    print(f"Evaluating metrics on {len(val_images)} validation images...")

    for img_name in val_images:
        img_path = os.path.join(val_dir, img_name)
        img_data = df[df['image'] == img_name]
        
        # Run YOLO inference
        results = model(img_path, verbose=False)[0]
        
        # Format ground truth data
        gt_list = [{'cx': r['center_x'], 'cy': r['center_y'], 'angle': r['angle_deg'], 'matched': False} 
                   for _, r in img_data.iterrows()]
            
        # Process network predictions
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            pred_kpts = results.keypoints.xy.cpu().numpy() 
            
            for kpt in pred_kpts:
                joint_x, joint_y = kpt[0]
                tab_x, tab_y = kpt[1]
                
                best_match, min_dist = None, float('inf')
                
                # Match prediction to ground truth using Euclidean distance threshold (20 pixels)
                for gt in gt_list:
                    if not gt['matched']:
                        dist = np.sqrt((joint_x - gt['cx'])**2 + (joint_y - gt['cy'])**2)
                        if dist < 20 and dist < min_dist:
                            min_dist, best_match = dist, gt
                            
                if best_match is not None:
                    best_match['matched'] = True
                    tp += 1
                    
                    # Calculate angle using inverse tangent (arctan2)
                    dx = tab_x - joint_x
                    dy = joint_y - tab_y # Y-axis is inverted in image coordinates
                    pred_angle = np.degrees(np.arctan2(dy, dx)) % 360
                    
                    # Calculate shortest rotational distance error
                    gt_angle = best_match['angle']
                    diff = abs(pred_angle - gt_angle)
                    angle_errors.append(min(diff, 360 - diff))
                else:
                    fp += 1 
                    
        # Count unmatched ground truths as False Negatives
        for gt in gt_list:
            if not gt['matched']: fn += 1

    # Calculate final aggregate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mae_angle = np.mean(angle_errors) if len(angle_errors) > 0 else 0

    print("\n" + "="*50)
    print(" FINAL VALIDATION METRICS ".center(50))
    print("="*50)
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1_score:.4f}")
    print(f"Angle Error (MAE): {mae_angle:.2f} degrees")
    print("="*50)

if __name__ == "__main__":
    evaluate()