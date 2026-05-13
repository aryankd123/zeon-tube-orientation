# 🧪 Tube Orientation Detection using Keypoint Estimation

This repository contains the solution for the machine learning assessment involving the detection of medical tubes and calculating their precise rotation angles from top-down RGB images. 

##  Architectural Approach

Instead of framing rotation prediction as a complex bounding-box regression problem—which often struggles to generalize on extremely small datasets—this solution utilizes a **Keypoint Detection (Pose Estimation)** pipeline.

1. **Detection:** A `YOLOv8n-pose` model was fine-tuned to identify two critical spatial landmarks per tube:
   * **Keypoint 1:** The central joint of the tube.
   * **Keypoint 2:** The tip of the alignment tab.
2. **Mathematical Conversion:** The predicted Cartesian coordinates are subsequently passed through a trigonometric function to reliably extract the orientation angle $\theta$:
   $$\theta = \arctan2(y_{joint} - y_{tab}, x_{tab} - x_{joint}) \pmod{360}$$

This hybrid approach (Deep Learning for spatial localization + Trigonometry for orientation) ensures mathematical rigidity and prevents the model from blindly "memorizing" angles.

##  Dataset & Evaluation Setup

* **Dataset:** 70 top-down images of a tube rack.
* **Split Strategy:** To provide honest, real-world metrics, the dataset was strictly split into an **80% Training Set** (56 images) and a **20% Validation Set** (14 images). 
* The model was evaluated *exclusively* on unseen data.

##  Final Validation Metrics

Despite the highly constrained training volume (56 images), the pipeline achieved highly generalized performance on the validation set:

| Metric | Score |
| :--- | :--- |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1 Score** | 1.0000 |
| **Mean Absolute Angle Error (MAE)** | **10.30 Degrees** |

*Note: The 10.30° error variance proves the network is actively predicting spatial coordinates dynamically rather than overfitting to the training distribution.*

##  How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt