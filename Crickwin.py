import streamlit as st
import pandas as pd
import joblib
import os

# ===========================
# 1. Load model safely with caching
# ===========================
MODEL_PATH = "live_probability_model_comp.joblib"

@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please upload or set correct path.")
        st.stop()
    try:
        return joblib.load(path)
    except AttributeError as e:
        st.error(f"Failed to load model due to missing class/module: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        st.stop()

model = load_model(MODEL_PATH)
st.success("✅ Model loaded successfully")

# ===========================
# 2. Page configuration & CSS
# ===========================
st.set_page_config(page_title="Cricket Chase Predictor", page_icon="🏏", layout="centered")

st.markdown("""
<style>
.stProgress > div > div > div > div { background-color: #2e8b57; }
.analysis-box { background-color: #f0f9f9; border-left: 6px solid #2e8b57; padding: 15px; margin-top: 15px; border-radius: 8px; }
.badge { display: inline-block; padding: 5px 10px; border-radius: 12px; font-weight: bold; color: white; }
.badge-green { background-color: #28a745; }
.badge-red { background-color: #dc3545; }
.badge-blue { background-color: #007bff; }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Cricket Match Win Probability Predictor")
st.subheader("Enter Match Details")

# ===========================
# 3. User inputs
# ===========================
col1, col2 = st.columns(2)

with col1:
    venue = st.selectbox("Venue", options=[
        'SuperSport Park', 'Dubai International Cricket Stadium', "Lord's", 'Others'
    ])
    innings = st.radio("Innings", options=[1, 2])

with col2:
    current_runs = st.number_input("Current Runs", min_value=0, value=0, step=1)
    wickets_in_hand = st.slider("Wickets in Hand", min_value=0, max_value=10, value=10, step=1)

overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
st.progress(min(overs_completed / 50, 1.0))

if innings == 2:
    target_score = st.number_input("Target Score", min_value=1, value=1, step=1)
    if current_runs > target_score:
        st.error("Current runs cannot exceed target for second innings!")
        st.stop()

# ===========================
# 4. Feature calculations
# ===========================
TOTAL_OVERS = 50
balls_bowled = overs_completed * 6
balls_remaining = max((TOTAL_OVERS * 6) - balls_bowled, 1)  # avoid division by zero

crr = current_runs / overs_completed if overs_completed > 0 else 0

# Prepare input data (match your trained model columns!)
input_data = pd.DataFrame({
    "Venue": [venue],
    "Current Score": [current_runs],
    "Wickets_in_Hand": [wickets_in_hand],
    "Overs Completed": [overs_completed],
    "CRR": [crr],
    "Innings": [innings],
    "RRR": [0],
    "Runs_to_Get": [0],
    "Balls_Remaining": [0],
    "Pressure": [0],
    "Home_Advantage": [0],  # default
})

if innings == 2:
    runs_to_get = target_score - current_runs
    rrr = (runs_to_get * 6) / balls_remaining
    pressure = runs_to_get / (wickets_in_hand + 1) if wickets_in_hand > 0 else runs_to_get
    input_data["RRR"] = [rrr]
    input_data["Runs_to_Get"] = [runs_to_get]
    input_data["Balls_Remaining"] = [balls_remaining]
    input_data["Pressure"] = [pressure]
else:
    projected_score = int(crr * TOTAL_OVERS)

# ===========================
# 5. Display live stats
# ===========================
st.markdown("### 📊 Live Match Stats")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"<span class='badge badge-blue'>CRR: {crr:.2f}</span>", unsafe_allow_html=True)
with col_b:
    if innings == 2:
        st.markdown(f"<span class='badge badge-red'>RRR: {rrr:.2f}</span>", unsafe_allow_html=True)
with col_c:
    st.markdown(f"<span class='badge badge-green'>Wickets: {wickets_in_hand}</span>", unsafe_allow_html=True)

# ===========================
# 6. Predict probability & analysis
# ===========================
if st.button("Predict Probability"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"✅ Probability of successful chase: {proba*100:.2f}%")

        if innings == 1:
            st.info(f"Projected Score at end of innings: {projected_score}")

        # Match situation analysis
        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
        st.markdown("### 🧐 Match Situation Analysis")
        if innings == 2:
            if proba > 0.75:
                st.write("The chasing team is **in complete control**! ✅")
            elif proba > 0.5:
                st.write("The chasing team has **the upper hand**, but needs to maintain momentum. 🔥")
            elif proba > 0.25:
                st.write("The match is **in the balance**. One good partnership can change everything. 🤔")
            else:
                st.write("The chasing team is **under serious pressure**! 😨")
        else:
            if projected_score >= 300:
                st.write("The batting team is on course for a **massive total**! 💥")
            elif projected_score >= 250:
                st.write("The batting team is heading for a **competitive total**. 🏏")
            else:
                st.write("The batting team needs to accelerate, or risk a low score. ⚠️")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error predicting: {e}")
