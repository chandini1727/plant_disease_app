# Plant Disease Detection App

A Deep Learning-based Web Application that classifies plant leaf diseases in crops like Pepper, Potato, and Tomato using MobileNetV2.  
The system provides real-time disease diagnosis, including causes and treatment recommendations — empowering farmers and agronomists with instant, AI-driven insights.

**Live Demo:** [Plant Disease App](https://plant-disease-app-hvgs.onrender.com)

---

## Problem Understanding

Agriculture is the backbone of our economy, yet plant diseases cause billions in losses yearly.  
Traditional methods are slow, expensive, and require expert intervention.

### Problem Statement

Develop a deep learning model capable of classifying plant leaf images into 15 categories (healthy + diseased), providing:
- Instant disease identification  
- Information on cause (fungal, bacterial, viral)  
- Suggested treatments  

---

## Functional Requirements

1. **Image Upload Interface**
   - Accepts JPG, PNG, or WebP files.
2. **Model Inference**
   - Classifies images using trained MobileNetV2 model.
3. **Prediction Display**
   - Shows predicted disease name, confidence, and treatment steps.
4. **Responsive UI**
   - Works seamlessly on desktop and mobile.
5. **Error Handling**
   - Handles invalid file types, missing inputs, and server errors.

---

## Non-Functional Requirements

| Category | Description |
|-----------|--------------|
| **Performance** | Fast inference using pre-trained MobileNetV2 |
| **Scalability** | Deployable on Render with Flask |
| **Usability** | Simple upload → result flow |
| **Reliability** | Automatic error handling and safe file cleanup |
| **Security** | Validates uploaded images and restricts file types |
| **Maintainability** | Modular code with `utils1.py` and clean routes |

---

## Components Overview

- **Frontend (Flask Templates):** Handles image uploads and displays predictions.
- **Backend (Flask App):**
  - Loads trained MobileNetV2 model.
  - Preprocesses images.
  - Predicts and maps output to disease details.
- **Model:**  
  - `plant_disease_mobileNetv2.h5` (15-class CNN model using transfer learning)
- **Data Source:**
  - Publicly available PlantVillage dataset (structured per crop & disease).
- **Deployment:**
  - Hosted on [Render](https://plant-disease-app-hvgs.onrender.com).


## 💻 Tech Stack

| Layer             | Technology                          |
| ----------------- | ----------------------------------- |
| **Frontend**      | HTML, CSS, Jinja2 (Flask Templates) |
| **Backend**       | Python (Flask Framework)            |
| **Model**         | TensorFlow / Keras (MobileNetV2)    |
| **Data Handling** | NumPy, Pillow, ImageDataGenerator   |
| **Deployment**    | Render Cloud                        |
| **Logging**       | Python Logging Module               |

---

## 🧠 Model Details

* **Base Model:** MobileNetV2 (`imagenet` weights)
* **Layers Added:**

  * `GlobalAveragePooling2D`
  * `Dropout`
  * `Dense` (ReLU)
  * `Softmax` for final classification
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Epochs:** 10
* **Accuracy:** ~97% on validation data

---

## 🧾 Example Disease Output

```json
{
  "Tomato___Early_blight": {
    "crop": "Tomato",
    "disease": "Early Blight",
    "confidence": "40.36%",
    "cause": "Caused by Alternaria fungi species. Shows characteristic concentric rings on lower leaves and progresses upward through the plant.",
    "treatment": "Remove infected leaves immediately. Use fungicide sprays such as chlorothalonil or mancozeb. Practice 3–4 year crop rotation, mulch around plants to prevent soil splash, and ensure proper plant spacing."
  }
}

```

##  Future Enhancements

* Integration with camera input for real-time detection
* Multilingual treatment suggestions for farmers
* Add more crops and disease datasets
* Convert to mobile app (Flutter + Flask API)

