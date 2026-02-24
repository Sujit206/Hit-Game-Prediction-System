# import streamlit as st
# import pandas as pd
# import joblib

# #load model
# model = joblib.load("hit_game_model.pkl")
# model_columns = joblib.load("hit_game_model.pkl")

# THRESHOLD = 0.7

# st.title("🎮 Hit Game Prediction System")

# #user Inputs
# platform = st.selectbox("Select Platform",["PS2","X360","PS4","DS"])

# genre = st.selectbox("Select Genre",["Action","Sports","Shooter","RPG"])

# publisher = st.selectbox("Select Publisher",["Nintendo","EA","Ubisoft","Activision"])

# year = st.slider("Release Year",1980,2020,2010)

# #Create input dataframe
# input_df = pd.DataFrame({
#     "platform":[platform],
#     "genre":[genre],
#     "publisher":[publisher],
#     "year":[year]
# })

# #One-hot encode same as training
# input_df = pd.get_dummies(input_df)

# #Align columns
# input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# if st.button("Predict"):
#     prob = model.predict_proba(input_df)[:, 1][0]
#     prediction = 1 if prob > 0.7 else 0

#     if prediction == 1:
#         st.success(f"Likely Hit!\nProbability: {prob:.2f}")
#     else:
#         st.warning(f"Not Likely Hit.\nProbability:{prob:.2f}")

import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load model and columns
# -------------------------
model = joblib.load("hit_game_model.pkl")
model_columns = joblib.load("model_columns.pkl")

THRESHOLD = 0.7

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Hit Game Predictor", page_icon="🎮")

st.title("🎮 Hit Game Classification System")
st.write("Predict whether a video game is likely to become a Hit based on historical data.")

st.markdown("---")

# -------------------------
# User Inputs
# -------------------------
platform = st.text_input("Platform (Example: PS2, X360, PS4, DS)")
genre = st.text_input("Genre (Example: Action, Sports, Shooter, RPG)")
publisher = st.text_input("Publisher (Example: Nintendo, EA, Ubisoft, Activision)")
year = st.number_input("Release Year", min_value=1980, max_value=2025, value=2010)

st.markdown("---")

if st.button("Predict Hit Probability"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "platform": [platform],
        "genre": [genre],
        "publisher": [publisher],
        "year": [year]
    })

    # One-hot encode
    input_data = pd.get_dummies(input_data)

    # Align columns with training data
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict probability
    probability = model.predict_proba(input_data)[:, 1][0]
    prediction = 1 if probability > THRESHOLD else 0

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"🔥 Likely HIT Game!\n\nProbability: {probability:.2f}")
    else:
        st.warning(f"⚠ Not Likely to be a Hit.\n\nProbability: {probability:.2f}")

    st.progress(float(probability))

st.markdown("---")
st.caption("Model: Logistic Regression | ROC-AUC: 0.81 | Threshold: 0.7")