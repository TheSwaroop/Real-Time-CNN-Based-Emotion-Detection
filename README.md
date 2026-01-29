# 🧠 Real-Time Emotion Detection System  
**CNN-Based Facial Emotion Recognition with Web Deployment**

---

## 📌 Overview  
This project is a **real-time facial emotion detection system** that uses a **Convolutional Neural Network (CNN)** to classify human emotions from images and live webcam video. The system integrates **machine learning, computer vision, and web deployment** into a single end-to-end pipeline, demonstrating how raw visual data can be preprocessed, analyzed, and served through an interactive analytics interface.

The application supports both **image upload** and **live video inference** and provides **emotion-based recommendations** (quotes, activities, and music) based on predicted emotional states.

---

## 🚀 Features  
- **CNN-based emotion classification** using TensorFlow/Keras  
- **Real-time webcam emotion detection** with OpenCV  
- **Image upload-based emotion inference**  
- **End-to-end data preprocessing pipeline** (grayscale conversion, resizing, normalization, tensor reshaping)  
- **Web-based interface** built with Streamlit  
- **Cloud-based model artifact loading** (model downloaded dynamically at runtime)  
- **Emotion-based recommendations** (quotes, activities, and music links)  

---

## 🏗️ System Architecture  
**High-Level Flow:**  
1. User uploads an image or starts webcam feed  
2. Frames/images are captured using OpenCV  
3. Data preprocessing pipeline prepares inputs for the CNN model  
4. CNN model performs real-time emotion classification  
5. Predictions are displayed in the Streamlit web interface  
6. Emotion-based recommendations are generated for the user  

---

## 🧠 Machine Learning Approach  
- **Model Type:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  
- **Input:** 48×48 grayscale facial images  
- **Output Classes:**  
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral  

### Data Preprocessing Pipeline  
- Convert RGB images to grayscale  
- Resize frames to 48×48 pixels  
- Normalize pixel values to the `[0,1]` range  
- Reshape into tensor format for CNN inference  

### Model Evaluation  
- Performance evaluated using **accuracy and loss metrics**  
- Architecture and hyperparameters iteratively tuned to improve classification reliability  

---

## 🌐 Web Application  
The application is built using **Streamlit**, allowing users to interact with the model through a browser-based interface.

### Supported Inputs  
- 📷 **Image Upload (JPG, PNG, JPEG)**  
- 🎥 **Live Webcam Feed (Real-Time Prediction)**  

---

## ☁️ Model Deployment & Artifact Management  
- The trained CNN model is stored in **cloud storage (Google Drive)**  
- The application dynamically downloads the model at runtime using `gdown`  
- This simulates a **cloud-based model deployment workflow**, similar to how ML artifacts are managed in production data platforms  

---

## 🛠️ Tech Stack  
**Programming Language:**  
- Python  

**Machine Learning & Data:**  
- TensorFlow / Keras  
- NumPy  
- OpenCV  

**Web Framework:**  
- Streamlit  

**Utilities:**  
- PIL (Image Processing)  
- gdown (Cloud Model Download)  

---

## 📂 Project Structure  
```text
Real-Time-Emotion-Detection/
│
├── model/
│   └── my_model.keras              # Trained CNN model (downloaded at runtime)
│
├── app.py                         # Main Streamlit application
├── RealTimeEmotionDetection.ipynb # Model training and experimentation notebook
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
