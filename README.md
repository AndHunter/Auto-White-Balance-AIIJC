# 📸 White Balance in Mobile Cameras

## 🏆 Competition Result

This project was developed as part of the **AIIJC (Artificial Intelligence & Image Intelligence Joint Challenge)**.

* **Track**: School Track
* **Result**: 🥉 **3rd place(https://aiijc.com/ru/results2025?tab=schoolchildren&direction=%D0%98%D0%BD%D1%81%D1%82%D0%B8%D1%82%D1%83%D1%82+AIRI)**
* **Competition website**: [https://aiijc.com/ru/](https://aiijc.com/ru/)

📦 **Dataset (Kaggle)**:
[https://www.kaggle.com/datasets/andrewsokolovsky/aiijc-abb3](https://www.kaggle.com/datasets/andrewsokolovsky/aiijc-abb3)

---

## Project Overview

Automatic White Balance (AWB) is a critical component of modern mobile camera pipelines.
Its goal is to ensure **color constancy** — the ability to perceive object colors consistently under different lighting conditions.

Most classical AWB algorithms assume the presence of a **single dominant light source**.
However, real-world analysis shows that **more than 50% of scenes contain multiple illuminants**, which makes such assumptions invalid.

This project addresses AWB as a **distribution prediction problem**, where the model predicts a **2D chromaticity histogram (128×128)** representing possible white points in the scene.

---

## Problem Statement

**Goal**:
Predict the **distribution of white points** in a scene while accounting for:

* multiple light sources,
* indoor / outdoor conditions,
* illumination type,
* camera metadata.

**Evaluation metric**:
**2D Wasserstein distance** between predicted and ground-truth histograms (lower is better).

---

## Proposed Approach

The solution follows a **multimodal, multi-task deep learning approach** that combines image content, metadata, and auxiliary supervision.

### Key Ideas

* 🔹 **Two separate models** for indoor and outdoor scenes
* 🔹 **Exposure correction** using `LightValue`
* 🔹 **Histogram parameterization via Gaussian Mixture Model (GMM)**

  * 16 components
  * 96 parameters instead of 16,384 bins
* 🔹 **Multi-task learning**

  * histogram regression
  * illumination type classification (`light_type`)

---

## Feature Representation

The model integrates multiple complementary cues:

* RGB image (Vision Transformer backbone)
* Log-chroma histogram (illumination-invariant)
* Edge map (Canny)
* Depth map (MiDaS-small)
* Brightness and saturation (HSV)
* Patch statistics (4×4 grid: mean, variance, entropy)

All features are concatenated into a **1728-dimensional vector**.

---

## Model Architecture

* **Backbone**: `vit_huge_patch14_224` (ImageNet pretrained)
* **Additional encoders**:

  * chroma histogram encoder
  * edge encoder
  * depth encoder
  * patch statistics encoder
  * brightness / saturation encoder
* **Outputs**:

  * GMM parameters → reconstructed 128×128 histogram
  * auxiliary classification head for `light_type` (8 classes)

### Loss Function

```
Loss = 0.7 × Sliced Wasserstein Distance
     + 0.3 × KL Divergence
     + 0.1 × CrossEntropyLoss(light_type)
```

This formulation improves convergence and encourages the model to understand scene illumination.

---

## Project structure

```
├── preprocessing.py      # Data preprocessing and feature extraction
├── train.py              # Model training
├── test.py               # Inference and submission generation
├── train_imgs/
├── test_imgs/
├── train_histograms/
├── train_content_markup/
├── test_content_markup/
└── README.md
```

---

## Installation

Python 3.8+ is recommended. GPU is strongly advised.

```bash
pip install torch torchvision timm opencv-python pandas rich POT lime
```

---

## Data Preprocessing

The project uses the following **correct directory configuration** in `preprocessing.py`:

```python
from pathlib import Path

DATA_DIR = Path("path_to_dataset")

TRAIN_IMGS_DIR = DATA_DIR / "train_imgs" / "train_imgs"
TRAIN_HISTS_DIR = DATA_DIR / "train_histograms" / "train_histograms"
TEST_IMGS_DIR = DATA_DIR / "test_imgs" / "test_imgs"

TRAIN_MARKUP_DIR = Path("train_content_markup/train_content_markup")
TEST_MARKUP_DIR = Path("test_content_markup/test_content_markup")
```

---

## How to Run

### Preprocessing

```bash
python preprocessing.py
```

### Training

```bash
python train.py
```

### Inference

```bash
python test.py
```

The output will be saved as `submission.zip`.

---

## Results

* 📊 Mean validation Wasserstein distance: **0.05 ± 0.02**
* 📈 Leaderboard score: **0.443**
* ✅ Significant improvement over baseline methods
* ✅ Strong impact of metadata-aware modeling
* ✅ Stable predictions across multiple random seeds

---

## Author

* **Telegram**: @main4562
* **X**: @main4562
* **Author**: Naymushin Andrey
---

## License

This project is released under the **MIT License**.
