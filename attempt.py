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


###2nd attempt
# import streamlit as st
# import pandas as pd
# import joblib

# # -------------------------
# # Load model and columns
# # -------------------------
# model = joblib.load("hit_game_model.pkl")
# model_columns = joblib.load("model_columns.pkl")

# THRESHOLD = 0.7

# # -------------------------
# # Page Config
# # -------------------------
# st.set_page_config(page_title="Hit Game Predictor", page_icon="🎮")

# st.title("🎮 Hit Game Classification System")
# st.write("Predict whether a video game is likely to become a Hit based on historical data.")

# st.markdown("---")

# # -------------------------
# # User Inputs
# # -------------------------
# platform = st.text_input("Platform (Example: PS2, X360, PS4, DS)")
# genre = st.text_input("Genre (Example: Action, Sports, Shooter, RPG)")
# publisher = st.text_input("Publisher (Example: Nintendo, EA, Ubisoft, Activision)")
# year = st.number_input("Release Year", min_value=1980, max_value=2025, value=2010)

# st.markdown("---")

# if st.button("Predict Hit Probability"):

#     # Create input dataframe
#     input_data = pd.DataFrame({
#         "platform": [platform],
#         "genre": [genre],
#         "publisher": [publisher],
#         "year": [year]
#     })

#     # One-hot encode
#     # input_data = pd.get_dummies(input_data)

#     # Align columns with training data
#     # input_data = input_data.reindex(columns=model_columns, fill_value=0)

#     #load columns
#     model_columns = joblib.load("model_columns.pkl")
    
#     #one-hot encoding
#     input_data = pd.get_dummies(input_data)

#     #add missing columns
#     for col in model_columns:
#         if col not in input_data.columns:
#             input_data[col] = 0

#     #keep only training columns and order
#     input_data = input_data[model_columns]

#     # Predict probability
#     probability = model.predict_proba(input_data)[:, 1][0]
#     prediction = 1 if probability > THRESHOLD else 0

#     st.subheader("Prediction Result")

#     if prediction == 1:
#         st.success(f"🔥 Likely HIT Game!\n\nProbability: {probability:.2f}")
#     else:
#         st.warning(f"⚠ Not Likely to be a Hit.\n\nProbability: {probability:.2f}")

#     st.progress(float(probability))

# st.markdown("---")
# st.caption("Model: Logistic Regression | ROC-AUC: 0.81 | Threshold: 0.7")


####### 3rd attempt #####

# import streamlit as st
# import pandas as pd
# import joblib

# # -----------------------------
# # CONFIG
# # -----------------------------
# THRESHOLD = 0.7

# st.set_page_config(page_title="Hit Game Predictor", page_icon="🎮")

# st.title("🎮 Hit Game Classification System")
# st.write("Predict whether a video game is likely to become a Hit.")

# st.markdown("---")

# # -----------------------------
# # LOAD MODEL & COLUMNS
# # -----------------------------
# try:
#     model = joblib.load("hit_game_model.pkl")
#     model_columns = joblib.load("model_columns.pkl")
# except Exception as e:
#     st.error("Model files not found. Make sure .pkl files are in the same folder.")
#     st.stop()

# # Extract unique values from model columns to get valid options
# platforms = sorted(list(set([col.replace("platform_", "") for col in model_columns if col.startswith("platform_")])))
# genres = sorted(list(set([col.replace("genre_", "") for col in model_columns if col.startswith("genre_")])))
# publishers = sorted(list(set([col.replace("publisher_", "") for col in model_columns if col.startswith("publisher_")])))

# # -----------------------------
# # USER INPUT
# # -----------------------------
# platform = st.selectbox("Platform", platforms)
# genre = st.selectbox("Genre", genres)
# publisher = st.selectbox("Publisher", publishers)
# year = st.number_input("Release Year", min_value=1980, max_value=2025, value=2010)

# st.markdown("---")

# # -----------------------------
# # PREDICTION BUTTON
# # -----------------------------
# if st.button("Predict"):

#     # Create dataframe
#     input_data = pd.DataFrame({
#         "platform": [platform],
#         "genre": [genre],
#         "publisher": [publisher],
#         "year": [year]
#     })

#     # One-hot encode
#     input_data = pd.get_dummies(input_data)

#     # Get expected feature names from the model
#     expected_columns = model.feature_names_in_
    
#     # Align columns with training data using reindex
#     input_data = input_data.reindex(columns=expected_columns, fill_value=0)

#     # -----------------------------
#     # PREDICT
#     # -----------------------------
#     try:
#         probability = model.predict_proba(input_data)[:, 1][0]
#         prediction = 1 if probability > THRESHOLD else 0

#         st.subheader("Prediction Result")

#         if prediction == 1:
#             st.success(f"🔥 Likely HIT Game!\n\nProbability: {probability:.2f}")
#         else:
#             st.warning(f"⚠ Not Likely to be a Hit.\n\nProbability: {probability:.2f}")

#         st.progress(float(probability))

#     except Exception as e:
#         st.error("Prediction failed. Please check input values.")
#         st.write(e)

# st.markdown("---")
# st.caption("Model: Logistic Regression | ROC-AUC: 0.81 | Threshold: 0.7")

"""<style> 
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(135deg, #2d1b69 0%, #0f0f1e 100%);
#     position: relative;
#     overflow: hidden;
# }
# [data-testid="stAppViewContainer"]::before {
#     content: '';
#     position: absolute;
#     top: -50%;
#     left: -50%;
#     width: 200%;
#     height: 200%;
#     background: radial-gradient(circle, rgba(147,51,234,0.15) 0%, transparent 70%), 
#                 radial-gradient(circle at right, rgba(196,181,253,0.12) 0%, transparent 70%),
#                 radial-gradient(circle at bottom, rgba(167,139,250,0.1) 0%, transparent 70%);
#     animation: containerPulse 8s ease-in-out infinite;
#     pointer-events: none;
# }
# @keyframes containerPulse {
#     0%, 100% { 
#         transform: scale(1) rotate(0deg);
#         opacity: 0.5;
#     }
#     25% { 
#         transform: scale(1.1) rotate(90deg);
#         opacity: 0.7;
#     }
#     50% { 
#         transform: scale(1.2) rotate(180deg);
#         opacity: 0.3;
#     }
#     75% { 
#         transform: scale(1.1) rotate(270deg);
#         opacity: 0.6;
#     }
# }
# [data-testid="stAppViewContainer"]::after {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: -100%;
#     width: 100%;
#     height: 100%;
#     background: linear-gradient(90deg, transparent, rgba(255,255,255,0.02), transparent);
#     animation: containerShimmer 6s infinite;
#     pointer-events: none;
# }
# @keyframes containerShimmer {
#     0% { left: -100%; }
#     100% { left: 100%; }
# }
# [data-testid="stSidebar"] {
#     background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
#     border-right: 2px solid #2d3561;
# }


</style>"""