import streamlit as st
import pandas as pd
import joblib
import os

# Load model
MODEL_PATH = "live_probability_model_comp.joblib"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please upload or set correct path.")
    st.stop()
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully")
except AttributeError as e:
    st.error(f"Failed to load model due to missing class/module: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading model: {e}")
    st.stop()

# Page configuration
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

# Title
st.title("🏏 Cricket Match Win Probability Predictor")

# User inputs
st.subheader("Enter Match Details")

col1, col2 = st.columns(2)
with col1:
    venue = st.selectbox("Venue", options=[
        'SuperSport Park', 'Dubai International Cricket Stadium', 'Happy Valley Ground', 'Meersen',
        'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)', "Lord's", 'Gahanga International Cricket Stadium. Rwanda',
        'Coolidge Cricket Ground', 'Bellerive Oval', 'King George V Sports Ground',
        'Pallekele International Cricket Stadium', 'Kerava National Cricket Ground', 'R Premadasa Stadium',
        'The Village', 'Sharjah Cricket Stadium', 'Amini Park', 'Shere Bangla National Stadium', 'Civil Service Cricket Club',
        'Marsa Sports Club', 'Gaddafi Stadium', 'Grange Cricket Club Ground', 'La Manga Club Bottom Ground',
        'Integrated Polytechnic Regional Centre', 'Sydney Cricket Ground', 'UKM-YSD Cricket Oval', 'Hagley Oval',
        'The Rose Bowl', 'ICC Academy', 'Saurashtra Cricket Association Stadium', 'Sportpark Het Schootsveld',
        'Independence Park', 'Rajiv Gandhi International Cricket Stadium', 'Wankhede Stadium',
        'Pierre Werner Cricket Ground', 'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)',
        'Sir Vivian Richards Stadium', 'Beausejour Stadium', 'Arun Jaitley Stadium', 'Kensington Oval', 'Manuka Oval',
        'Hurlingham Club Ground', 'Goldenacre', 'Moara Vlasiei Cricket Ground', 'OUTsurance Oval', 'Sophia Gardens',
        'Zayed Cricket Stadium', 'Adelaide Oval', 'Vidarbha Cricket Association Stadium',
        'Bayer Uerdingen Cricket Ground', 'Trent Bridge', 'Sheikh Abu Naser Stadium', 'Newlands',
        'JSCA International Stadium Complex', 'Old Trafford', 'Greenfield International Stadium', 'Holkar Cricket Stadium',
        'Bayuemas Oval', 'MA Chidambaram Stadium', 'Providence Stadium', 'Mission Road Ground', 'Sheikh Zayed Stadium',
        'Harare Sports Club', 'Gymkhana Club Ground', 'National Stadium', 'Indian Association Ground', 'Arnos Vale Ground',
        'Bulawayo Athletic Club', 'New Wanderers Stadium', 'Eden Park', 'Svanholm Park', 'County Ground',
        'Tribhuvan University International Cricket Ground', 'Kinrara Academy Oval', 'St Albans Club',
        'M.Chinnaswamy Stadium', 'Riverside Ground', 'Europa Sports Complex', 'Terdthai Cricket Ground',
        'Western Australia Cricket Association Ground', 'Desert Springs Cricket Ground',
        'Sano International Cricket Ground', 'Willowmoore Park', 'Tikkurila Cricket Ground', 'VRA Ground', 'Kingsmead',
        'Eden Gardens', 'McLean Park', 'National Cricket Stadium', 'Kennington Oval', 'Seddon Park',
        'Royal Brussels Cricket Club Ground', 'Rawalpindi Cricket Stadium', 'Khan Shaheb Osman Ali Stadium',
        'Gahanga International Cricket Stadium', 'National Sports Academy', 'Entebbe Cricket Oval', 'GMHBA Stadium',
        'Lisicji Jarak Cricket Ground', "St George's Park", 'ICC Academy Ground No 2',
        'Mahinda Rajapaksa International Cricket Stadium', 'White Hill Field', 'Scott Page Field', 'Warner Park',
        'Punjab Cricket Association Stadium', 'Sylhet Stadium', 'Melbourne Cricket Ground', 'Tolerance Oval',
        'The Wanderers Stadium', 'Maharashtra Cricket Association Stadium', 'Rajiv Gandhi International Stadium',
        'Queens Sports Club', "Queen's Park Oval", 'Punjab Cricket Association IS Bindra Stadium', 'GB Oval',
        'Westpac Stadium', 'West End Park International Cricket Stadium', 'Central Broward Regional Park Stadium Turf Ground',
        'M Chinnaswamy Stadium', 'Belgrano Athletic Club Ground', 'Hazelaarweg', 'Castle Avenue', 'Wanderers Cricket Ground',
        'Stadium Australia', 'Windsor Park', 'Bready Cricket Club', 'Sportpark Westvliet', 'Perth Stadium', 'Senwes Park',
        'Zahur Ahmed Chowdhury Stadium', 'ICC Global Cricket Academy', 'Mombasa Sports Club Ground', 'Green Park',
        'Subrata Roy Sahara Stadium', 'Bay Oval', 'Boland Park', 'Edgbaston', 'Himachal Pradesh Cricket Association Stadium',
        'Narendra Modi Stadium', 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
        'United Cricket Club Ground', 'University of Lagos Cricket Oval', 'R.Premadasa Stadium', 'Brisbane Cricket Ground',
        'Buffalo Park', 'Carrara Oval', 'Kyambogo Cricket Oval', 'Sawai Mansingh Stadium',
        'Greater Noida Sports Complex Ground', 'Malahide', 'Sabina Park', 'Barabati Stadium', 'College Field', 'Headingley',
        'P Sara Oval', 'Gucherre Cricket Ground', 'Tony Ireland Stadium', 'Sylhet International Cricket Stadium',
        'Darren Sammy National Cricket Stadium', 'Grange Cricket Club', 'Lugogo Cricket Oval', 'Simonds Stadium',
        'Feroz Shah Kotla', 'Solvangs Park', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'AMI Stadium',
        'Moses Mabhida Stadium', 'Santarem Cricket Ground', 'Mangaung Oval', 'Sardar Patel Stadium', 'John Davies Oval',
        'Sportpark Maarschalkerweerd', 'Wanderers', 'University Oval', 'Jade Stadium', 'Brian Lara Stadium',
        'Clontarf Cricket Club Ground', 'Brabourne Stadium', 'Sky Stadium', 'Maple Leaf North-West Ground',
        'Barsapara Cricket Stadium', 'Saxton Oval', 'De Beers Diamond Oval', "Others"
    ])
    innings = st.radio("Innings", options=[1, 2])

with col2:
    current_runs = st.number_input("Current Runs", min_value=0, value=0, step=1)
    wickets_in_hand = st.slider("Wickets in Hand", min_value=0, max_value=10, value=10, step=1)

# Overs input with progress bar
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
st.progress(min(overs_completed / 50, 1.0))

# Only ask for target in 2nd innings
if innings == 2:
    target_score = st.number_input("Target Score", min_value=1, value=1, step=1)
    if current_runs > target_score:
        st.error("Current runs cannot exceed target for second innings!")
        st.stop()

# Calculate dynamic features
TOTAL_OVERS = 50
balls_bowled = overs_completed * 6
balls_remaining = (TOTAL_OVERS * 6) - balls_bowled

# Current Run Rate
crr = current_runs / overs_completed if overs_completed > 0 else 0

# Prepare input data with all training features
input_data = pd.DataFrame({
    "Venue": [venue],
    "Country": ["Unknown"],  # Default value
    "Pitch Conditions": ["Average"],  # Default value
    "Current Score": [current_runs],
    "Wickets Fallen": [10 - wickets_in_hand],  # Calculated from Wickets_in_Hand
    "Overs Completed": [overs_completed],
    "CRR": [crr],
    "Home_Advantage": [0],  # Default value
    "Innings_Phase": ["Middle"],  # Default value
    "Wickets_in_Hand": [wickets_in_hand],
    "Runs_Last_5_Overs": [0],  # Default value (could be calculated if data available)
    "Wickets_Last_5_Overs": [0],  # Default value
    "Weighted_Wickets": [0],  # Default value
    "Pressure": [0],  # Will be updated below
    "Innings": [innings],
    "RRR": [0],
    "Runs_to_Get": [0],
    "Balls_Remaining": [0],
})

# Update dynamic features for 2nd innings
if innings == 2:
    runs_to_get = target_score - current_runs
    rrr = (runs_to_get * 6) / balls_remaining if balls_remaining > 0 else 0
    pressure = runs_to_get / (wickets_in_hand + 1) if wickets_in_hand > 0 else 0
    input_data["RRR"] = [rrr]
    input_data["Runs_to_Get"] = [runs_to_get]
    input_data["Balls_Remaining"] = [balls_remaining]
    input_data["Pressure"] = [pressure]
else:
    projected_score = int(crr * TOTAL_OVERS)
    input_data["Pressure"] = [0]  # Default for 1st innings

# Display live stats
st.markdown("### 📊 Live Match Stats")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"<span class='badge badge-blue'>CRR: {crr:.2f}</span>", unsafe_allow_html=True)
with col_b:
    if innings == 2:
        st.markdown(f"<span class='badge badge-red'>RRR: {rrr:.2f}</span>", unsafe_allow_html=True)
with col_c:
    st.markdown(f"<span class='badge badge-green'>Wickets: {wickets_in_hand}</span>", unsafe_allow_html=True)

# Predict probability
if st.button("Predict Probability"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        st.success(f"✅ Probability of successful chase: {proba*100:.2f}%")

        if innings == 1:
            st.info(f"Projected Score at end of innings: {projected_score}")

        # Match Situation Analysis
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
