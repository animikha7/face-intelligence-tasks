# Task A: Gender Classification using EfficientNet-B4

This task focuses on building a robust binary classifier to distinguish between **male** and **female** faces using the **EfficientNet-B4** deep learning architecture.

---

## 📁 Dataset Structure

The dataset should follow the structure:

Task_A/
├── male/
└── female/


Each subfolder contains face images of the corresponding gender.

---

## 🧠 Model Overview

- **Architecture**: Pretrained `EfficientNet-B4` from `efficientnet_pytorch`
- **Modification**: Final layer replaced to output 2 logits (male/female)
- **Training**:
  - Optimizer: Adam
  - Loss: CrossEntropyLoss with class weighting
  - Precision: Mixed Precision (AMP)
- **Class Imbalance**: Handled using `WeightedRandomSampler` based on label distribution

---

## 🔄 Training Pipeline

1. **Image Preprocessing**
   - Resize to 380×380
   - Random horizontal flip, rotation, color jitter
   - Normalization (ImageNet stats)

2. **Data Loading**
   - Split into training and validation sets
   - Apply `WeightedRandomSampler` to balance gender classes

3. **Training Loop**
   - Train model using AMP (autocast + GradScaler)
   - Track training and validation accuracy
   - Save best model weights

4. **Evaluation**
   - Generate classification report (precision, recall, F1)
   - Visualize confusion matrix

---

## 📊 Results

After training, the model is evaluated on the validation/test set with:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

All metrics are printed and plotted directly in the notebook.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch
- torchvision
- efficientnet_pytorch
- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm

---

## 🚀 How To Run

1. Upload your dataset to Google Drive in the required structure
2. Open and run `Task_A.ipynb` in Google Colab
3. Set the dataset path and execute all cells
4. Model will be trained, validated, and saved with evaluation metrics

---

## ✅ Key Features

- AMP-enabled fast training on GPU
- Handles imbalanced datasets efficiently
- Accurate and lightweight EfficientNet-B4 backbone
- Clean, modular, and easy-to-adapt code

---

## 📁 Output

- Trained model weights stored in your Google Drive
- Classification report and confusion matrix visuals

---

## 🔄 Next Steps

- Try with other EfficientNet variants (e.g., B0, B3, B7)
- Fine-tune on your own dataset
- Integrate into mobile or web applications

---

## 🧑‍💻 Author

This solution was developed for the **COMSYS Hackathon 5 - Task A** using modern deep learning best practices.
