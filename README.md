
# PestDetectionCV

**PestDetectionCV** is a computer vision project focused on automated detection and classification of insect pests using deep learning object detection models. The aim of this work is to support precision agriculture by enabling scalable pest identification from images, reducing reliance on manual inspection, and demonstrating expertise in state-of-the-art detection techniques.

This project includes implementations and experiments for:

- A **Faster R-CNN** based detector with MobileNetV3 backbone and dropout regularization
- A **YOLOv8n** based detector with benchmarking, noise robustness testing, and performance evaluation

This repository documents data preparation, model training, evaluation, inference, and visualization workflows suitable for research, practical deployment, and demonstration purposes.

---

## 1. Project Overview

Manual inspection of pest damage on crops is slow and resource intensive. Automated pest detection using computer vision can significantly accelerate monitoring by identifying and localizing pests in images.

**PestDetectionCV** implements and evaluates two detection paradigms:

- A **region-based detection model (Faster R-CNN)** optimized for accuracy
- A **single-pass detection model (YOLOv8n)** optimized for speed and robustness

This repository includes training scripts, evaluation notebooks, inference scripts, and demo utilities.

---

## 2. Motivation and Objectives

The primary goals of this project are:

- To apply deep learning object detection methods to a real-world agricultural problem
- To benchmark and compare performance under different noise, augmentation, and resolution conditions
- To build end-to-end workflows for training, testing, and deploying detection models
- To demonstrate technical proficiency in dataset preprocessing, model training, experiment design, and performance analysis

---

## 3. Technical Approach

This project uses established object detection frameworks in combination with custom training and evaluation workflows:

- **Faster R-CNN** (with MobileNetV3) for accurate bounding box predictions
- **YOLOv8n** for real-time detection and experimental benchmarking
- Noise robustness and augmentation experiments to test model resilience
- Evaluation across precision, recall, mean Average Precision (mAP), and detection time

Training is performed primarily in **Google Colab** for GPU acceleration, with dataset and checkpoints stored in **Google Drive**.

---

## 4. Dataset Structure

The dataset used across both models follows a directory hierarchy compatible with YOLO and custom loaders:


dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/


- Images are saved in JPEG or PNG format
- Labels are in a text format compatible with YOLO or converted to Faster R-CNN input formats

---

## 5. Model Descriptions

### Faster R-CNN

Faster R-CNN is a two-stage object detector consisting of:

- A **Region Proposal Network (RPN)** that suggests candidate object regions
- A **classification and bounding box refinement head**

This implementation uses **MobileNetV3** as the backbone and includes dropout regularization to reduce overfitting. The workflow supports custom collation and training scripts.

**Key files:**

- `train_dropout.py` or `train_model.py`
- `demo.py` for inference visualization

### YOLOv8n

YOLOv8n is a lightweight object detector designed for fast inference. This project contains multiple experimental workflows written as independent notebooks:

- `yolov8n_benchmark_EDA.ipynb` – Baseline performance and exploratory data analysis
- `yolov8n_hyperparameter_testing.ipynb` – Tests mosaic augmentation, noise corruption, and resolution changes
- `yolov8n_models_optimised_eval.ipynb` – Evaluation of optimized models
- `yolov8n_demo.ipynb` – Inference demonstration and visualizations

Dependencies include the **Ultralytics YOLO** library and common image processing packages.

---

## 6. Google Colab Training Workflows

### Faster R-CNN Training

1. Upload `dataset.zip` to Google Drive
2. Mount Drive in Colab
3. Extract data to Colab workspace
4. Upload and run `train_dropout.py`

**Typical result:**

- Model saved as `faster_rcnn_mobilenet_AdamW30_dropout.pth`
- Training logs include epoch loss, mAP @ 0.5, and class-wise AP

### YOLOv8n Training and Experiments

In Colab, install dependencies and run each notebook in sequence. Each notebook is independent and performs a specific experimental task such as noise robustness, hyperparameter exploration, or final evaluation.

---

## 7. Local Inference and Demo

After training, run models locally for inference.

### Faster R-CNN Demo

**Project directory:**


project_directory/
├── demo.py
├── faster_rcnn_mobilenet_AdamW30_dropout.pth
├── test/
    ├── images/
    └── labels/


**Install dependencies:**

```bash
pip install torch torchvision pillow opencv-python numpy
```

**Run the demo:**

```bash
python demo.py
```

---

## 8. Results and Evaluation Metrics

Evaluation metrics used across models include:

- Mean Average Precision (mAP@0.5)
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score
- Inference time per image

Results and performance graphs are generated in the YOLOv8n notebooks and logged in the Colab console for Faster R-CNN.

---

## 10. Directory Structure

```
PestDetectionCV/
├── RCNN/              # Faster R-CNN training and inference scripts
├── YOLO/              # YOLOv8n notebooks and experimental code
├── Report/            # Written project report
├── README.md          # This document
└── requirements.txt   # Python dependencies (if provided)
```

---

## 11. How to Run

1. Clone the repository
2. Prepare dataset in the required structure
3. Select model (Faster R-CNN or YOLOv8n)
4. For Colab training, upload necessary scripts
5. Run training and evaluation
6. Perform inference using saved weights


