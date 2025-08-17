import streamlit as st
import pandas as pd
import joblib
import os

# ===========================
# 1. Load your trained model
# ===========================
MODEL_PATH = "model_clean_without_target_comp.joblib"  # adjust path
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload or set correct path.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Cricket Chase Predictor", page_icon="🏏", layout="centered")

# Custom CSS for cricket feel
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #2e8b57;
    }
    .analysis-box {
        background-color: #f0f9f9;
        border-left: 6px solid #2e8b57;
        padding: 15px;
        margin-top: 15px;
        border-radius: 8px;
    }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 12px;
        font-weight: bold;
        color: white;
    }
    .badge-green { background-color: #28a745; }
    .badge-red { background-color: #dc3545; }
    .badge-blue { background-color: #007bff; }
    </style>
""", unsafe_allow_html=True)

st.title("🏏 Cricket Match Win Probability Predictor")

# ===========================
# 2. User inputs
# ===========================
st.subheader("Enter Match Details")

col1, col2 = st.columns(2)
with col1:
    venue = st.selectbox("Venue", options=['SuperSport Park', 'Dubai International Cricket Stadium',
       'Happy Valley Ground', 'Meersen',
       'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)',
       "Lord's", 'Gahanga International Cricket Stadium. Rwanda',
       'Coolidge Cricket Ground', 'Bellerive Oval',
       'King George V Sports Ground',
       'Pallekele International Cricket Stadium',
       'Kerava National Cricket Ground', 'R Premadasa Stadium',
       'The Village', 'Sharjah Cricket Stadium', 'Amini Park',
       'Shere Bangla National Stadium', 'Civil Service Cricket Club',
       'Marsa Sports Club', 'Gaddafi Stadium',
       'Grange Cricket Club Ground', 'La Manga Club Bottom Ground',
       'Integrated Polytechnic Regional Centre', 'Sydney Cricket Ground',
       'UKM-YSD Cricket Oval', 'Hagley Oval', 'The Rose Bowl',
       'ICC Academy', 'Saurashtra Cricket Association Stadium',
       'Sportpark Het Schootsveld', 'Independence Park',
       'Rajiv Gandhi International Cricket Stadium', 'Wankhede Stadium',
       'Pierre Werner Cricket Ground',
       'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)',
       'Sir Vivian Richards Stadium', 'Beausejour Stadium',
       'Arun Jaitley Stadium', 'Kensington Oval', 'Manuka Oval',
       'Hurlingham Club Ground', 'Goldenacre',
       'Moara Vlasiei Cricket Ground', 'OUTsurance Oval',
       'Sophia Gardens', 'Zayed Cricket Stadium', 'Adelaide Oval',
       'Vidarbha Cricket Association Stadium',
       'Bayer Uerdingen Cricket Ground', 'Trent Bridge',
       'Sheikh Abu Naser Stadium', 'Newlands',
       'JSCA International Stadium Complex', 'Old Trafford',
       'Greenfield International Stadium', 'Holkar Cricket Stadium',
       'Bayuemas Oval', 'MA Chidambaram Stadium', 'Providence Stadium',
       'Mission Road Ground', 'Sheikh Zayed Stadium',
       'Harare Sports Club', 'Gymkhana Club Ground', 'National Stadium',
       'Indian Association Ground', 'Arnos Vale Ground',
       'Bulawayo Athletic Club', 'New Wanderers Stadium', 'Eden Park',
       'Svanholm Park', 'County Ground',
       'Tribhuvan University International Cricket Ground',
       'Kinrara Academy Oval', 'St Albans Club', 'M.Chinnaswamy Stadium',
       'Riverside Ground', 'Europa Sports Complex',
       'Terdthai Cricket Ground',
       'Western Australia Cricket Association Ground',
       'Desert Springs Cricket Ground',
       'Sano International Cricket Ground', 'Willowmoore Park',
       'Tikkurila Cricket Ground', 'VRA Ground', 'Kingsmead',
       'Eden Gardens', 'McLean Park', 'National Cricket Stadium',
       'Kennington Oval', 'Seddon Park',
       'Royal Brussels Cricket Club Ground', 'Rawalpindi Cricket Stadium',
       'Khan Shaheb Osman Ali Stadium',
       'Gahanga International Cricket Stadium', 'National Sports Academy',
       'Entebbe Cricket Oval', 'GMHBA Stadium',
       'Lisicji Jarak Cricket Ground', "St George's Park",
       'ICC Academy Ground No 2',
       'Mahinda Rajapaksa International Cricket Stadium',
       'White Hill Field', 'Scott Page Field', 'Warner Park',
       'Punjab Cricket Association Stadium', 'Sylhet Stadium',
       'Melbourne Cricket Ground', 'Tolerance Oval',
       'The Wanderers Stadium', 'Maharashtra Cricket Association Stadium',
       'Rajiv Gandhi International Stadium', 'Queens Sports Club',
       "Queen's Park Oval",
       'Punjab Cricket Association IS Bindra Stadium', 'GB Oval',
       'Westpac Stadium', 'West End Park International Cricket Stadium',
       'Central Broward Regional Park Stadium Turf Ground',
       'M Chinnaswamy Stadium', 'Belgrano Athletic Club Ground',
       'Hazelaarweg', 'Castle Avenue', 'Wanderers Cricket Ground',
       'Stadium Australia', 'Windsor Park', 'Bready Cricket Club',
       'Sportpark Westvliet', 'Perth Stadium', 'Senwes Park',
       'Zahur Ahmed Chowdhury Stadium', 'ICC Global Cricket Academy',
       'Mombasa Sports Club Ground', 'Green Park',
       'Subrata Roy Sahara Stadium', 'Bay Oval', 'Boland Park',
       'Edgbaston', 'Himachal Pradesh Cricket Association Stadium',
       'Narendra Modi Stadium',
       'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
       'United Cricket Club Ground', 'University of Lagos Cricket Oval',
       'R.Premadasa Stadium', 'Brisbane Cricket Ground', 'Buffalo Park',
       'Carrara Oval', 'Kyambogo Cricket Oval', 'Sawai Mansingh Stadium',
       'Greater Noida Sports Complex Ground', 'Malahide', 'Sabina Park',
       'Barabati Stadium', 'College Field', 'Headingley', 'P Sara Oval',
       'Gucherre Cricket Ground', 'Tony Ireland Stadium',
       'Sylhet International Cricket Stadium',
       'Darren Sammy National Cricket Stadium', 'Grange Cricket Club',
       'Lugogo Cricket Oval', 'Simonds Stadium', 'Feroz Shah Kotla',
       'Solvangs Park',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'AMI Stadium', 'Moses Mabhida Stadium', 'Santarem Cricket Ground',
       'Mangaung Oval', 'Sardar Patel Stadium', 'John Davies Oval',
       'Sportpark Maarschalkerweerd', 'Wanderers', 'University Oval',
       'Jade Stadium', 'Brian Lara Stadium',
       'Clontarf Cricket Club Ground', 'Brabourne Stadium', 'Sky Stadium',
       'Maple Leaf North-West Ground', 'Barsapara Cricket Stadium',
       'Saxton Oval', 'De Beers Diamond Oval']

)
    innings = st.radio("Innings", options=[1, 2])

with col2:
    current_runs = st.number_input("Current Runs", min_value=0, value=0, step=1)
    # 🔹 Moving bar for wickets
    wickets_in_hand = st.slider("Wickets in Hand", min_value=0, max_value=10, value=10, step=1)

# Overs input with progress bar
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
st.progress(min(overs_completed / 50, 1.0))  # Overs progress bar

# Only ask for target in 2nd innings
if innings == 2:
    target_score = st.number_input("Target Score", min_value=1, value=1, step=1)
    if current_runs > target_score:
        st.error("Current runs cannot exceed target for second innings!")
        st.stop()

# ===========================
# 3. Prepare input DataFrame
# ===========================
input_data = pd.DataFrame({
    "Venue": [venue],
    "Current Score": [current_runs],
    "Wickets_in_Hand": [wickets_in_hand],
    "Overs Completed": [overs_completed],
    "Innings": [innings],
    "CRR": [0],
    "RRR": [0],
    "Runs_to_Get": [0],
    "Balls_Remaining": [0],
    "Pressure": [0],
    "Home_Advantage": [0],
})

# ===========================
# 4. Calculate features dynamically
# ===========================
TOTAL_OVERS = 50
balls_bowled = overs_completed * 6
balls_remaining = (TOTAL_OVERS * 6) - balls_bowled

# Current Run Rate
crr = current_runs / overs_completed if overs_completed > 0 else 0
input_data["CRR"] = [crr]

projected_score = None
if innings == 1:
    projected_score = int(crr * TOTAL_OVERS)
    input_data["Projected_Score"] = [projected_score]
else:
    runs_to_get = target_score - current_runs
    rrr = (runs_to_get * 6) / balls_remaining if balls_remaining > 0 else 0
    pressure = runs_to_get / (wickets_in_hand + 1)
    input_data["Runs_to_Get"] = [runs_to_get]
    input_data["Balls_Remaining"] = [balls_remaining]
    input_data["RRR"] = [rrr]
    input_data["Pressure"] = [pressure]

# ===========================
# 5. Display live stats (CRR, RRR)
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
# 6. Predict probability
# ===========================
if st.button("Predict Probability"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"✅ Probability of successful chase: {proba*100:.2f}%")

        if innings == 1:
            st.info(f"Projected Score at end of innings: {projected_score}")

        # ===========================
        # 7. Match Situation Analysis
        # ===========================
        st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
        st.markdown("### 🧐 Match Situation Analysis")
        if innings == 2:
            if proba > 0.75:
                st.write("The chasing team is **in complete control**! A win looks almost certain. ✅")
            elif proba > 0.5:
                st.write("The chasing team has **the upper hand**, but needs to maintain the momentum. 🔥")
            elif proba > 0.25:
                st.write("The match is **in the balance**. One good partnership can change everything. 🤔")
            else:
                st.write("The chasing team is **under serious pressure**! Needs a miracle. 😨")
        else:
            if projected_score >= 300:
                st.write("The batting team is on course for a **massive total**! Bowlers beware. 💥")
            elif projected_score >= 250:
                st.write("The batting team is heading for a **competitive total**. Game on! 🏏")
            else:
                st.write("The batting team needs to accelerate, or they risk a low score. ⚠️")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error predicting: {e}")
