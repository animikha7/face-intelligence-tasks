# Task B: Face Recognition Workflow

This workflow describes each major step of the face recognition pipeline using Facenet-PyTorch, including data handling, feature extraction, matching, and evaluation.

---

## 1. Setup & Dependencies

- Install required packages and import libraries.
- Set device (GPU/CPU).

## 2. Data and Model Preparation

- Define dataset directories (`train_dir`, `val_dir`).
- Load pretrained FaceNet (InceptionResnetV1) and MTCNN models.

## 3. Embedding Extraction with TTA

- For each image, apply test-time augmentation (brightness, sharpness, contrast, original).
- Detect and align faces with MTCNN.
- Extract FaceNet embeddings and average them.

## 4. Build Validation Feature Database

- For each person in `val_dir`, extract embeddings from clean images.
- Store embeddings per person.

## 5. Distorted Image Matching

- For each distorted image, extract embedding.
- Compute cosine similarity with all validation embeddings.
- Find the most similar person.

## 6. Thresholding and Result Storage

- Accept a match if similarity exceeds the threshold.
- Store results (file, true label, predicted label, score, match).

## 7. Evaluation and Analysis

- Print sample results.
- Compute Top-1 Accuracy and Macro F1 Score.
- Plot threshold tuning and similarity histograms.

## 8. Visualization

- Threshold tuning curve.
- Histogram of similarity scores.

## 9. End
