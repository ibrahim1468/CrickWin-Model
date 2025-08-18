import streamlit as st
import pandas as pd
import joblib
import os
import gdown

st.set_page_config(page_title="Cricket Chase Predictor", page_icon="🏏", layout="centered")

# ===========================
# 1. Load Model (cached, lazy)
# ===========================

MODEL_PATH = "live_probability_model_comp.joblib"
FILE_ID = "1k1YdOdPzg9eyGUObTSDyFwNxsz9Urp7m"  # <-- only the ID
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model (~250MB)... please wait ⏳"):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🏏 Cricket Match Win Probability Predictor")

# ===========================
# 2. User inputs
# ===========================
st.subheader("Enter Match Details")

venue = st.selectbox("Venue", options=[
    'SuperSport Park': , 'Dubai International Cricket Stadium',
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
       'Saxton Oval', 'De Beers Diamond Oval', 'Others'
])

innings = st.radio("Innings", options=[1, 2])

current_runs = st.number_input("Current Runs", min_value=0, value=0, step=1)
wickets_in_hand = st.number_input("Wickets in Hand", min_value=0, max_value=10, value=10, step=1)
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

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
# 4. Calculate features
# ===========================
TOTAL_OVERS = 50
balls_bowled = overs_completed * 6
balls_remaining = (TOTAL_OVERS * 6) - balls_bowled

if overs_completed > 0:
    crr = current_runs / overs_completed
else:
    crr = 0
input_data["CRR"] = [crr]

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
# 5. Predict probability
# ===========================
if st.button("Predict Probability"):
    try:
        with st.spinner("Running prediction..."):
            proba = model.predict_proba(input_data)[0][1]
        st.success(f"✅ Probability of successful chase: {proba*100:.2f}%")
        if innings == 1:
            st.info(f"Projected Score at end of innings: {projected_score}")
    except Exception as e:
        st.error(f"Error predicting: {e}")

