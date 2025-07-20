# 🌿 Plant Disease Detection App

A deep learning-based mobile/web-compatible system to classify plant leaf diseases across crops like Pepper, Potato, and Tomato using MobileNetV2. Upon prediction, the system displays detailed diagnostic reports including disease cause and treatment recommendations.

---

## 🧠 Problem Statement

Develop a deep learning model that can classify plant leaf images into 15 specific categories, including both healthy and diseased conditions. The goal is to:

- Accurately identify disease types in crops such as Pepper, Potato, and Tomato.
- Provide users (especially farmers) with real-time diagnostic insights and treatment recommendations.

---

## 🔍 Project Overview

### ✅ Why CNNs in Agriculture?
Convolutional Neural Networks (CNNs) have proven effective in analyzing visual data. This project demonstrates how CNNs can automate the detection of plant diseases with high accuracy, offering quick insights to farmers and agricultural experts.

### ⚙️ What We Built:
- **Model**: MobileNetV2 (with transfer learning)
- **Classes**: 15 categories including healthy and diseased leaf images
- **Platform**: Mobile or Web (compatible)
- **Output**: Predicted class, cause, and recommended treatment

---

## 📁 Dataset Overview

- Directory structured with 15 subfolders (each representing one class)
- Example categories:
  - `Pepper__bell___healthy`, `Pepper__bell___Bacterial_spot`
  - `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
  - `Tomato___Leaf_mold`, `Tomato___YellowLeafCurlVirus`, `Tomato___healthy`, etc.

---

## 🚀 Procedure

### 🔹 1. Data Preparation
- Used `ImageDataGenerator` with:
  - Rescaling
  - Rotation
  - Shearing
  - Zoom
  - Horizontal flipping
- Split into training and validation datasets

### 🔹 2. Model Architecture
- **Base**: MobileNetV2 (`include_top=False`, `weights='imagenet'`)
- **Custom Layers**:
  - `GlobalAveragePooling2D`
  - `Dropout`
  - `Dense` layers with ReLU
  - Final `Dense` layer with Softmax (for 15-class classification)
- **Compilation**:
  - Optimizer: `Adam`
  - Loss: `categorical_crossentropy`
  - Metric: `accuracy`

### 🔹 3. Model Training
- Trained for 10 epochs
- Used validation set for monitoring performance and avoiding overfitting

### 🔹 4. Model Saving
- Final model saved as: `plant_disease_mobileNetv2.h5`

---

## 📊 Model Evaluation

- Input: Path to a test leaf image
- Preprocessing: Resize to `224x224x3`, rescale, expand dimensions
- Output:
  - Predicted disease class
  - Confidence score
  - Complete disease diagnostic info (from a dictionary)

---

## 🧾 Disease Information Dictionary

Each class label (e.g., `Tomato___Late_blight`) is mapped to:

- **Crop**
- **Disease name**
- **Cause** (e.g., fungus, bacteria, virus)
- **Treatment** (e.g., fungicides, pruning, resistant seeds)

Example output:

```json
{
  "Tomato___Late_blight": {
    "crop": "Tomato",
    "disease": "Late Blight",
    "cause": "Fungus (Phytophthora infestans)",
    "treatment": "Use fungicides, remove infected leaves, ensure proper drainage."
  }
}
