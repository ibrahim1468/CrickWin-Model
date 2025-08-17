import streamlit as st
import joblib
import os
import gdown

# Google Drive link
FILE_ID = "1k1YdOdPzg9eyGUObTSDyFwNxsz9Urp7m"  # extracted from your link
MODEL_PATH = "live_probability_model_comp.joblib"

@st.cache_data(show_spinner=True, persist=True)
def load_model():
    # Download the model if not present
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    # Load the model
    return joblib.load(MODEL_PATH)

# Load model safely
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

