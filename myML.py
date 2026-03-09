import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# CONFIG
# -----------------------------
THRESHOLD = 0.7

st.set_page_config(page_title="Hit Game Predictor", page_icon="🎮", layout="wide")

# Custom CSS for aesthetic design
st.markdown("""
<style>

    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    h1 {
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: titleGlow 3s ease-in-out infinite alternate;
    }

    h2{
        font-size: 1.3rem !important;
        font-weight: 600 !important;

    }

    .h2Glow {
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        # -webkit-text-fill-color: transparent;
        background-clip: text;
        # font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    .pGlow {
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        # font-size: 3.5rem !important;
        font-weight: 600 !important;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 20px rgba(255,255,255,0.5)); }
        to { filter: drop-shadow(0 0 30px rgba(255,255,255,0.8)); }
    }
    

    /* Floating particles effect */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        animation: float 6s infinite ease-in-out;
    }
    <!-- Floating particles -->
    <div class="particles">
        <div class="particle" style="width: 4px; height: 4px; left: 10%; animation-delay: 0s;"></div>
        <div class="particle" style="width: 6px; height: 6px; left: 20%; animation-delay: 1s;"></div>
        <div class="particle" style="width: 3px; height: 3px; left: 30%; animation-delay: 2s;"></div>
        <div class="particle" style="width: 5px; height: 5px; left: 40%; animation-delay: 3s;"></div>
        <div class="particle" style="width: 4px; height: 4px; left: 50%; animation-delay: 4s;"></div>
        <div class="particle" style="width: 6px; height: 6px; left: 60%; animation-delay: 5s;"></div>
        <div class="particle" style="width: 3px; height: 3px; left: 70%; animation-delay: 6s;"></div>
        <div class="particle" style="width: 5px; height: 5px; left: 80%; animation-delay: 7s;"></div>
        <div class="particle" style="width: 4px; height: 4px; left: 90%; animation-delay: 8s;"></div>
    </div>

    .main-header>h1{
       text-align: center;
    }
            
.main-header {
    position: relative;
    width: 100%;
    # background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    background: linear-gradient(45deg, #fff, #f0f0f0);
    -webkit-background-clip: text;
    padding: 2.8rem 0;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    text-align: center;
    # box-shadow: 0 15px 35px rgba(0,0,0,0.1), 0 5px 15px rgba(0,0,0,0.1);
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 25px;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.03), transparent);
    animation: shimmer 3s infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}


.hero-card {
    background: linear-gradient(135deg, #e94560 0%, #ff6b6b 50%, #4ecdc4 100%);
    padding: 2.5rem;
    border-radius: 25px;
    color: white;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.1) 100%);
    animation: heroShimmer 4s infinite;
}
@keyframes heroShimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.input-card {
    height: 240px;
    width: 100%;
    # background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    padding: 2.5rem;
    # border-radius: 25px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    margin-bottom: 2rem;
    border: 2px solid #2d3561;
    color: white;
    position: absolute;
    overflow: hidden;
    transition: all 0.3s ease;

    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 25px;
}
.input-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.2);
    border-color: #4ecdc4;
}
.input-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(78,205,196,0.2), transparent);
    transition: left 0.6s ease;
}
.input-card:hover::before {
    left: 100%;
}

div[data-baseweb="select"] {
    width: 90%;
    position: relative;
    margin-left: 2rem;
}


.result-card {
    padding: 2.5rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: slideIn 0.5s ease-out;

    background: rgba(255, 255, 255, 0.0);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 25px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);

}

    div[data-testid="stMetric"] {
        width: 90%;
        text-align: center;
        background: rgba(255, 255, 255, 0.0);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    }

    .metric-prob {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }



@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.success-result {
    background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #00d2ff 100%);
    color: white;
    border: 2px solid #4ecdc4;
}
.warning-result {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 50%, #ff6b6b 100%);
    color: white;
    border: 2px solid #e94560;
}
.metric-label {
    font-size: 1.4rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    color: #a8b2d1;

    backdrop-filter: blur(20px);
    border-radius: 25px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);

    background: linear-gradient(45deg, #fff, #f0f0f0);
    -webkit-background-clip: text;
    # -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    animation: titleGlow 3s ease-in-out infinite alternate;

}
.footer > p{
    line-height: 1.5;
    font-family: 'Inter', sans-serif;
    color: rgba(255, 255, 255, 0.9)


}

.stButton > button {
    background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
    # background: linear-gradient(135deg, #e94560 0%, #ff6b6b 50%, #4ecdc4 100%);
    color: white;
    border: none;
    padding: 1rem 3rem;
    font-weight: bold;
    font-size: 1.1rem;
    border-radius: 50px;
    transition: all 0.3s ease;
    box-shadow: 0 10px 25px rgba(78,205,196,0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.1) 100%);
    animation: heroShimmer 4s infinite;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 15px 35px rgba(78,205,196,0.5);
    # background: linear-gradient(135deg, #44a08d 0%, #4ecdc4 100%);
    # background: linear-gradient(135deg, #4ecdc4 0%, #ff6b6b 50%, #e94560 100%);
}
.stButton > button:active {
    transform: translateY(-1px) scale(1.03);
}
.stSelectbox > div > div {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #2d3561;
    border-radius: 15px;
    color: white;
}
.stNumberInput > div > div {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #2d3561;
    border-radius: 15px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎮 Hit Game Classification System</h1>
    <p class="pGlow">Advanced ML-powered prediction for gaming success</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <h2 class="h2Glow">Welcome to the Future of Game Prediction!</h2>
    <p class="pGlow">Our sophisticated machine learning model analyzes platform, genre, publisher, and release year to predict whether a game will become a hit. With 81% ROC-AUC accuracy, you're getting industry-leading insights.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# LOAD MODEL & COLUMNS
# -----------------------------
try:
    model = joblib.load("hit_game_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    st.error("Model files not found. Make sure .pkl files are in the same folder.")
    st.stop()

# Extract unique values from model columns to get valid options
platforms = sorted(list(set([col.replace("platform_", "") for col in model_columns if col.startswith("platform_")])))
genres = sorted(list(set([col.replace("genre_", "") for col in model_columns if col.startswith("genre_")])))
publishers = sorted(list(set([col.replace("publisher_", "") for col in model_columns if col.startswith("publisher_")])))

# -----------------------------
# USER INPUT
# -----------------------------
col1, col2= st.columns(2)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("🎯 Game Details")
    platform = st.selectbox("🎮 Platform", platforms)
    genre = st.selectbox("🎭 Genre", genres)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("🏢 Publishing Info")
    publisher = st.selectbox("📦 Publisher", publishers)
    # year = st.number_input("📅 Release Year", min_value=1980, max_value=2025, value=2010)
    year = st.selectbox("📅 Release Year", ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'])
    st.markdown('</div>', unsafe_allow_html=True)



# Center the predict button
st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
predict_button = st.button("🔮 Predict Game Success", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if predict_button:

    # Create dataframe
    input_data = pd.DataFrame({
        "platform": [platform],
        "genre": [genre],
        "publisher": [publisher],
        "year": [year]
    })

    # One-hot encode
    input_data = pd.get_dummies(input_data)

    # Get expected feature names from the model
    expected_columns = model.feature_names_in_
    
    # Align columns with training data using reindex
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # -----------------------------
    # PREDICT
    # -----------------------------
    try:
        probability = model.predict_proba(input_data)[:, 1][0]
        prediction = 1 if probability > THRESHOLD else 0

        # result_class = "success-result" if prediction == 1 else "warning-result"
        emoji = "🔥" if prediction == 1 else "⚠️"
        status = "HIT GAME!" if prediction == 1 else "NOT HIT"
        
        st.markdown(f"""
        <div class="result-card">
            <div class="inner">
            <h2>{emoji} Prediction: {status}</h2>
            <div class="metric-label">Success Probability: {probability:.2%}</div>
            <div style="margin: 1rem 0;">
                <div style="background: rgba(255,255,255,0.4); border-radius: 10px; padding: 0.5rem;">
                    <div style="background: rgba(255,255,255,0.4); height: 20px; border-radius: 8px; width: {probability*100}%;"></div>
                </div>
            </div>
            <p><strong>Decision Threshold:</strong> {THRESHOLD:.2f}</p>
            <p><strong>Verdict:</strong> {'This game shows strong potential for success!' if prediction == 1 else 'This game may need additional marketing support to succeed.'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative progress bar for compatibility
        # st.progress(float(probability))   
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-prob">Probability', unsafe_allow_html=True)
            st.metric("", f"{probability:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-prob">Status', unsafe_allow_html=True)
            st.metric("", "HIT" if prediction == 1 else "NOT HIT")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-prob">Confidence', unsafe_allow_html=True)
            st.metric("", "High" if probability > 0.8 else "Medium" if probability > 0.6 else "Low")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Prediction failed. Please check input values.")
        st.write(e)

st.markdown("""
<div class="footer">
    <p class="foot-parag"><strong>🤖 Model Information</strong></p>
    <p class="foot-parag">Logistic Regression | ROC-AUC: 0.81 | Decision Threshold: 0.7</p>
    <p class="foot-parag">Trained on historical video game data for accurate hit prediction</p>
    <p class="foot-parag" style="font-size: 0.8rem; margin-top: 1rem;">© 2026 Hit Game Predictor | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
