# Tube Orientation Detection with Keypoint Estimation

Detects tubes in top-down RGB images of a tube rack and estimates each tube's rotation angle. Built as a solution to a machine learning assessment.

## Approach

Rather than regressing rotated bounding boxes, which tends to generalise poorly on very small datasets, the model predicts two keypoints per tube and the angle is computed from them.

1. **Detection:** a `YOLOv8n-pose` model fine-tuned to find two keypoints per tube:
   * Keypoint 1: the central joint of the tube
   * Keypoint 2: the tip of the alignment tab
2. **Angle:** computed from the two keypoints:
   $$\theta = \arctan2(y_{joint} - y_{tab}, x_{tab} - x_{joint}) \pmod{360}$$

## Data and evaluation

* 70 top-down images of a tube rack
* 80/20 split: 56 training images, 14 validation images
* All metrics below are on the validation images only

## Results (validation set)

| Metric | Score |
| :--- | :--- |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 score | 1.00 |
| Mean absolute angle error | 10.30° |

Detection is perfect on this small validation set; angle accuracy is the main thing left to improve. With only 14 validation images, these numbers should be read as indicative.

## How to run

```bash
pip install -r requirements.txt
python 1_data_preparation.py   # build the keypoint dataset and train/validation split
python 2_train_model.py        # fine-tune YOLOv8n-pose
python 3_evaluate_model.py     # metrics on the validation set
```
