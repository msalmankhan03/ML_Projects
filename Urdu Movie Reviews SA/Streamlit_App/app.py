import streamlit as st
import joblib
import re
import os

# -------------------------- Safe File Paths --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sentiment_model.pkl").replace("\\", "/")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl").replace("\\", "/")

# -------------------------- Load Model & Vectorizer --------------------------
try:
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# -------------------------- Page Config --------------------------
st.set_page_config(
    page_title="Urdu Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -------------------------- CSS Styling --------------------------
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #FFD700;
    font-family: 'Cursive', sans-serif;
    font-size: 48px;
}
.stTextArea>div>div>textarea {
    font-size: 18px;
    font-family: 'Noto Nastaliq Urdu', serif;
}
.stButton>button {
    background-color: #FFD700;
    color: black;
    font-size: 18px;
    width: 100%;
    border-radius: 10px;
}
.stApp {
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
.result-card {
    padding: 20px;
    border-radius: 10px;
    font-size: 24px;
    text-align: center;
    margin-top: 20px;
}
.positive {
    background-color: #d4edda;
    color: #155724;
}
.negative {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- Background Image --------------------------
def add_bg_from_local(image_file="background.jpg"):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = f.read()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded.hex()});
        }}
        </style>
        """, unsafe_allow_html=True)

# Use your background image or default
add_bg_from_local("background.jpg")

# -------------------------- Title --------------------------
st.markdown("🎬 <h1>Urdu Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write(" ")

# -------------------------- Clean Text Function --------------------------
def clean_text(text):
    return re.sub(r"[^؀-ۿ\s]", "", text)

# -------------------------- Input & Prediction --------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_input = st.text_area("✍ Enter Urdu Movie Review:", height=150)

    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text!")
        else:
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            if prediction == "positive":
                st.markdown(f"<div class='result-card positive'>😃 Positive Review!</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card negative'>😞 Negative Review!</div>", unsafe_allow_html=True)
