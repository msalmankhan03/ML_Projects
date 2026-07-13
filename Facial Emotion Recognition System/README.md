# 🎭 AI Facial Intelligence Hub: Real-Time Mood Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

## 🌟 Overview
The **AI Facial Intelligence Hub** is a sophisticated Deep Learning application designed for real-time human emotion recognition. Built using **PyTorch** and **OpenCV**, this system can accurately identify seven primary human emotions from live webcam feeds or uploaded images. It features a professional-grade analysis dashboard tailored for high-precision mood assessment.

---

## 🚀 Key Features

### 🎥 Dual Detection Modes
- **Live Webcam Stream**: Real-time emotion tracking with low-latency inference.
- **Static Image Analysis**: Upload any portrait to extract detailed emotional data.

### 📊 Professional Analysis Dashboard
- **Stability-Based Detection**: An "Auto-Stop" feature that ensures emotional stability by analyzing 50 consecutive frames before finalizing a result.
- **Confidence Metrics**: Real-time bar charts showing probability distribution across all emotion categories.
- **Detailed Reports**: Generates a final summary including primary emotion, confidence scores, and stability verification.

### 🛠️ Robust Engineering
- **Corrupted Data Filtering**: Automated scripts to clean datasets from broken or non-image files.
- **Optimized Inference**: Uses Haar Cascades for rapid face localization and a custom CNN for precise classification.

---

## 🧠 Model Architecture: FacialExpressionCNN

The core of this project is a custom **Deep Convolutional Neural Network (CNN)** implemented in PyTorch, optimized for the FER-2013 dataset (48x48 grayscale images).

### Layer Breakdown:
1.  **Input Layer**: 48x48x1 (Grayscale)
2.  **Conv Block 1**: Two 3x3 Conv layers (32 & 64 filters) + BatchNorm + MaxPool + Dropout (25%)
3.  **Conv Block 2**: Two 3x3 Conv layers (128 filters) + BatchNorm + MaxPool + Dropout (25%)
4.  **Conv Block 3**: Two 3x3 Conv layers (256 filters) + BatchNorm + MaxPool + Dropout (25%)
5.  **Fully Connected**: 512 neurons + BatchNorm + Dropout (50%)
6.  **Output**: Softmax layer for 7 classes (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

---

## 📂 Project Workflow

### 1. Data Preprocessing (`scripts/`)
- **`clean_data.py`**: Scans the `data/` directory, verifies image integrity using PIL, and removes corrupted files to prevent training crashes.
- **`data_loader.py`**: Implements a robust PyTorch `DataLoader` with data augmentation (Random Horizontal Flip, Rotation) and normalization.

### 2. Model Implementation (`models/`)
- **`model.py`**: Defines the `FacialExpressionCNN` class architecture with residual-style depth and heavy regularization to prevent overfitting.

### 3. Verification (`verify_setup.py`)
- Checks environment compatibility, including GPU availability (CUDA), PyTorch installation, and OpenCV backends.

### 4. Real-Time Application (`app.py`)
- A **Streamlit** powered interface that integrates the trained model with OpenCV's webcam capture.
- Implements the **Stability Logic**:
    - Requires **50 frames** of continuous tracking.
    - Requires **>= 70%** confidence threshold.
    - Requires the most frequent emotion to appear in at least **35/50** frames.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam (for live mode)

### Steps
1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd Facial-Expression
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Environment**
   ```bash
   python verify_setup.py
   ```

4. **Launch the Hub**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Directory Structure
```text
├── data/               # Training and Testing datasets
├── models/             
│   └── model.py        # CNN Architecture
├── scripts/            
│   ├── clean_data.py   # Dataset cleaning utility
│   └── data_loader.py  # PyTorch data pipeline
├── outputs/            
│   └── best_model.pth  # Trained model weights
├── app.py              # Streamlit Frontend
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 🔮 Future Enhancements
- [ ] **Temporal Analysis**: Integrate LSTMs or Transformers to analyze emotion transitions over time.
- [ ] **Multi-Face Support**: Enable tracking for multiple people simultaneously in a frame.
- [ ] **Emotion-Driven UI**: Dynamically change application themes based on the detected mood.

---
*Created for the Facial Expression Recognition Project - 2024*
