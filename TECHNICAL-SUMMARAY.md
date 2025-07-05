
# Face-Intelligence-Tasks

This repository addresses two challenging computer vision problems using the **FACECOM** dataset:

- **Task A: Gender Classification**
- **Task B: Face Recognition with Distorted Images**

---

## 🧠 Task A: Gender Classification

### 🎯 Objective

To classify facial images as either **male** or **female**.

### ⚙️ Model & Techniques

- **Architecture**: Transfer learning using **EfficientNet-B0** via `efficientnet_pytorch`.
- **Augmentation**: Utilized `torchvision.transforms` for resizing, normalization, flipping, etc.
- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizer**: Adam optimizer with gradient scaling (`GradScaler`) for AMP (Automatic Mixed Precision)
- **Training Techniques**:
  - Weighted sampling to address class imbalance
  - Mixed precision training for speed and efficiency
  - Early stopping logic
- **Evaluation**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - Classification Report from `sklearn`

### 📂 Data Structure

```
Task_A/
├── train/
│   ├── male/
│   └── female/
└── val/
    ├── male/
    └── female/
```

---

## 🧑‍💻 Task B: Face Recognition with Distorted Images

### 🎯 Objective

To recognize and match distorted face images to their corresponding identities.

### ⚙️ Model & Approach

- **Architecture**: 
  - Used **EfficientNetV2-S** from `timm` as the backbone
  - Integrated with an **ArcFace head** for discriminative embeddings
- **Embedding Strategy**:
  - Anchor images (clean faces) and distorted versions are processed into embeddings
  - Compared via **cosine similarity**
- **Loss Function**: ArcMarginProduct + CrossEntropyLoss
- **Optimizer**: AdamW
- **Training Enhancements**:
  - Cosine Annealing LR scheduler
  - Label smoothing
- **Thresholding**:
  - Used cosine similarity histograms to tune decision thresholds for matching

### 📂 Data Structure

```
Task_B/
├── person1/
│   ├── img.jpg
│   └── distortion/
│       ├── aug1.jpg
│       └── aug2.jpg
├── person2/
│   ├── img.jpg
│   └── distortion/
        ...
```

### 📊 Evaluation

- Pair-wise identity match prediction
- Accuracy, Precision, Recall, F1-score computed using ground truth pairs
- Visualizations:
  - Histogram of cosine similarities
  - Confusion Matrix
  - ROC-like threshold plots

---

## ✅ Requirements

- Python 3.8+
- torch, torchvision, efficientnet_pytorch, timm
- scikit-learn, matplotlib, pandas, seaborn
- tqdm, PIL

---

## 📁 Project Structure

```
Comys_Hackathon5/
├── Task_A/
│   ├── train/
│   ├── val/
│   └── Task_A.ipynb
├── Task_B/
│   ├── train/
│   ├── val/
│   └── Task_B.ipynb
└── README.md
```

---

## 🚀 Conclusion

This repository showcases deep learning techniques for:
- Gender classification with class imbalance handling
- Face recognition under distortions using ArcFace

It provides modular, scalable, and reproducible notebooks ready for extension to real-world biometric pipelines.
