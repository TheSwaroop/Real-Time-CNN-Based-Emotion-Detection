# 🧠 Real-Time CNN-Based Emotion Detection System  
**Facial Emotion Recognition with Machine Learning, Computer Vision, and Web Deployment**

---

## 📌 Overview  
This project implements a **real-time facial emotion recognition system** using a **Convolutional Neural Network (CNN)** trained on the **FER2013 dataset**. The system classifies human emotions from both **static images and live webcam video** and provides **emotion-based recommendations** such as motivational quotes, activities, and music playlists to create an interactive and supportive user experience.

The solution demonstrates a full **end-to-end machine learning pipeline**, including **data preprocessing, model training, evaluation, deployment, and real-time inference** through a web-based interface.

---

## 🎯 Problem Statement  
Recognizing human emotions in real-time can enhance **mental health support, entertainment systems, and human–computer interaction**. Many traditional systems rely only on static images and do not provide meaningful feedback to users.

This project addresses these limitations by:
- Supporting **live webcam-based emotion detection**
- Using a **deep learning model (CNN)** instead of traditional classifiers
- Providing **personalized recommendations** based on detected emotional states

---

## 🚀 Features  
- **CNN-based emotion classification** using TensorFlow/Keras  
- **Live webcam emotion detection** with OpenCV  
- **Image upload-based emotion inference**  
- **End-to-end data preprocessing pipeline**  
- **Web-based user interface** built with Streamlit  
- **Cloud-based model artifact loading** (dynamic model download at runtime)  
- **Emotion-based recommendations** (quotes, activities, and music playlists)  
- **Multilingual playlist support** (English & Telugu)  

---

## 🏗️ System Architecture  
### High-Level Flow  
1. User uploads an image or starts webcam feed  
2. Frames/images are captured using OpenCV  
3. Data preprocessing pipeline prepares inputs for the CNN model  
4. CNN model performs emotion classification  
5. Predictions are displayed in the web interface  
6. Emotion-based recommendations are generated for the user  

---

## 📊 Dataset  
**Dataset Name:** FER2013 (Facial Expression Recognition 2013)  

- **Total Samples:** 35,887 images  
- **Image Size:** 48 × 48 (Grayscale)  
- **Classes (7):**  
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral  

### Dataset Structure  
Each row contains:  
- `emotion` → Label (0–6)  
- `pixels` → Space-separated pixel values (0–255)  
- `Usage` → Dataset split  
  - Training (80%)  
  - Validation / PublicTest (10%)  
  - Testing / PrivateTest (10%)  

---

## 🧠 Machine Learning Pipeline  

### Data Preprocessing  
- Detect and remove missing values  
- Convert pixel strings to NumPy arrays  
- Reshape images to `(48, 48, 1)`  
- Normalize pixel values to `[0, 1]`  
- One-hot encode emotion labels  

**Outputs:**  
`X_train, y_train, X_val, y_val, X_test, y_test`

---

### CNN Model Architecture  
- **Input Layer:**  
  - `Conv2D(64, (3,3), ReLU)` → Edge and feature detection  
- **Hidden Layers:**  
  - `Conv2D(128, (3,3), ReLU)`  
  - `BatchNormalization()`  
  - `MaxPooling2D(2,2)`  
  - `Flatten()`  
  - `Dense(512, ReLU)`  
  - `Dropout(0.5)`  
- **Output Layer:**  
  - `Dense(7, Softmax)` → Emotion probabilities  

---

### Model Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Batch Size:** 64  
- **Epochs:** 10  
- **Validation:** PublicTest split  

---

### Model Evaluation  
- Evaluated using **accuracy and loss metrics**  
- Achieved approximately **58% classification accuracy**  
- Iterative tuning applied to improve model reliability  

---

## 🌐 Web Application  
The system is deployed as a **Streamlit web application** that allows real-time interaction with the trained CNN model.

### Supported Inputs  
- 📷 **Image Upload (JPG, PNG, JPEG)**  
- 🎥 **Live Webcam Feed**  

### Output  
- Detected emotion label  
- Motivational quote  
- Suggested activity  
- Music playlist (English/Telugu)  

---

## ☁️ Model Deployment & Artifact Management  
- Trained model is stored in **cloud storage (Google Drive)**  
- Application dynamically downloads the model at runtime using `gdown`  
- This simulates a **cloud-based ML deployment workflow**, similar to enterprise data platforms  

---

## 🛠️ Tech Stack  
**Programming Language:**  
- Python  

**Machine Learning & Data:**  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Pandas  

**Web Framework:**  
- Streamlit  

**Utilities:**  
- PIL (Image Processing)  
- gdown (Cloud Model Download)  

---

## 📂 Project Structure  
```text
Real-Time-CNN-Based-Emotion-Detection/
│
├── model/
│   └── my_model.keras               # Trained CNN model (downloaded at runtime)
│
├── app.py                          # Streamlit web application
├── RealTimeEmotionDetection.ipynb # Model training and experimentation notebook
├── workflow.pdf                   # System workflow diagram
├── ML project.pptx               # Project presentation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
