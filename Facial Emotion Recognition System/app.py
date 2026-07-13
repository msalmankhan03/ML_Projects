import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torchvision import transforms
import pandas as pd
import time

# Project Setup
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from models.model import FacialExpressionCNN

st.set_page_config(page_title="AI Mood Analyzer", layout="wide")

@st.cache_resource
def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model = FacialExpressionCNN(num_classes=len(classes)).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'best_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, device, classes
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None, None, None

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model, device, classes = load_trained_model()
face_cascade = load_face_cascade()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("🎭 Facial Intelligence Hub")

# Sidebar - Settings and Mode
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Detection Mode", ["Live Webcam", "Image Upload"])
    auto_stop = st.checkbox("Auto-Stop on Conclusion", value=True, help="Stop detection once a stable emotion is identified")
    
    if st.button("Reset Detection"):
        if 'is_stopped' in st.session_state:
            st.session_state.is_stopped = False
        if 'emotion_history' in st.session_state:
            st.session_state.emotion_history = []
        st.rerun()

col1, col2 = st.columns([2, 1])

def process_and_display(frame):
    """Core detection logic shared by both modes"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Performance optimization: detect faces on a smaller gray image
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    faces = face_cascade.detectMultiScale(small_gray, 1.3, 5)

    probs = [0.0] * len(classes)
    label = "No Face Detected"
    current_conf = 0.0

    for (x, y, w, h) in faces:
        x, y, w, h = x*2, y*2, w*2, h*2
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0: continue
        
        roi_tensor = transform(roi).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(roi_tensor)
            p = F.softmax(out, dim=1)
            conf, pred = torch.max(p, 1)
            label = classes[pred.item()]
            probs = p.cpu().numpy()[0]
            current_conf = conf.item()
        
        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_rgb, f"{label.upper()} ({current_conf:.2f})", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame_rgb, label, probs, current_conf

def get_camera():
    if 'cap' in st.session_state and st.session_state.cap is not None:
        if st.session_state.cap.isOpened():
            return st.session_state.cap
        else:
            st.session_state.cap.release()
    
    for index in [0, 1, 2]:
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]:
            cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    st.session_state.cap = cap
                    return cap
                cap.release()
    return None

if model is None:
    st.error("Model file 'outputs/best_model.pth' not found or corrupted!")
else:
    if mode == "Live Webcam":
        with col1:
            run = st.checkbox('Start Live Stream', value=True)
            video_placeholder = st.empty()
            
        with col2:
            st.subheader("Real-time Stats")
            emotion_text = st.empty()
            chart_placeholder = st.empty()
            summary_placeholder = st.empty()

        if run and not st.session_state.get('is_stopped', False):
            camera = get_camera()
            if camera is None:
                st.error("Could not access any webcam.")
            else:
                if 'emotion_history' not in st.session_state:
                    st.session_state.emotion_history = []
                
                try:
                    while run:
                        ret, frame = camera.read()
                        if not ret: break

                        frame_rgb, label, probs, conf = process_and_display(frame)

                        # Stability Fix: Encode image to JPEG bytes to prevent MediaFileStorageError
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        frame_bytes = buffer.tobytes()

                        # Update UI
                        video_placeholder.image(frame_bytes, width="stretch")
                        emotion_text.metric("Current Emotion", label.upper(), delta=f"{conf:.1%}")
                        
                        chart_data = pd.DataFrame(probs, index=classes, columns=['Confidence'])
                        chart_placeholder.bar_chart(chart_data)

                        # Stability Check
                        if label != "No Face Detected":
                            st.session_state.emotion_history.append(label)
                            if len(st.session_state.emotion_history) > 50: # Increased to 50 frames (~2.5s)
                                st.session_state.emotion_history.pop(0)
                            
                            if len(st.session_state.emotion_history) == 50:
                                most_common = max(set(st.session_state.emotion_history), key=st.session_state.emotion_history.count)
                                count = st.session_state.emotion_history.count(most_common)
                                
                                # Professional Requirement: Stable (70% of 50 frames = 35) AND High Confidence (>= 70%)
                                if count > 35 and conf >= 0.70:
                                    # Save final results to session state
                                    st.session_state.final_emotion = most_common
                                    st.session_state.final_conf = conf
                                    st.session_state.final_probs = probs
                                    
                                    if auto_stop:
                                        st.session_state.is_stopped = True
                                        st.rerun()
                                        break
                        
                        time.sleep(0.07) 
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif st.session_state.get('is_stopped', False):
            # Professional Results Dashboard for Final Year Project
            st.markdown("---")
            st.header("🏁 Final Analysis Report")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.success(f"### Detected Primary Emotion: **{st.session_state.final_emotion.upper()}**")
                st.metric("Final Confidence Score", f"{st.session_state.final_conf:.2%}")
                st.info(f"Analysis completed based on stable feature extraction over 50 consecutive frames.")
            
            with res_col2:
                st.write("#### Confidence Distribution")
                final_chart = pd.DataFrame(st.session_state.final_probs, index=classes, columns=['Confidence'])
                st.bar_chart(final_chart)
            
            st.divider()
            st.warning("⏱️ Detection has been paused to preserve the final result. Click 'Reset Detection' in the sidebar to start a new analysis.")
    
    else: # Image Upload Mode
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                processed_img, label, probs, conf = process_and_display(img)
                st.image(processed_img, caption="Processed Image", width="stretch")
                
        with col2:
            if uploaded_file is not None:
                st.subheader("Analysis Results")
                st.metric("Detected Emotion", label.upper(), delta=f"{conf:.1%}")
                
                chart_data = pd.DataFrame(probs, index=classes, columns=['Confidence'])
                st.bar_chart(chart_data)
                
                if label != "No Face Detected":
                    st.success(f"**Detailed Summary:** The subject appears to be **{label.upper()}** with a confidence score of **{conf:.2%}**.")
                else:
                    st.warning("No face was clearly detected in the uploaded image.")


