import streamlit as st
import pandas as pd
import joblib

#load model
model = joblib.load("./hit_game_model.pkl")

st.title("🎮 Hit Game Prediction System")

#user Inputs
platform = st.selectbox("Select Platform",["PS2","X360","PS4","DS"])

genre = st.selectbox("Select Genre",["Action","Sports","Shooter","RPG"])

publisher = st.selectbox("Select Publisher",["Nintendo","EA","Ubisoft","Activision"])

year = st.slider("Release Year",1980,2020,2010)

#Create input dataframe
input_df = pd.DataFrame({
    "platform":[platform],
    "genre":[genre],
    "publisher":[publisher],
    "year":[year]
})

#One-hot encode same as training
input_df = pd.get_dummies(input_df)

#Align columns
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict"):
    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if prob > 0.7 else 0

    if prediction == 1:
        st.success(f"Likely Hit!\nProbability: {prob:.2f}")
    else:
        st.warning(f"Not Likely Hit.\nProbability:{prob:.2f}")