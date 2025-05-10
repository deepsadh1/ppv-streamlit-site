
import streamlit as st
import numpy as np
import joblib
from PIL import Image

model = joblib.load("xgboost_ppv_model_final.pkl")

rock_constants = {
    "Limestone": {"K": 105.60, "beta": 0.90, "Rock_Type_Limestone": 1},
    "Coal": {"K": 209.77, "beta": 1.22, "Rock_Type_Limestone": 0}
}

st.set_page_config(page_title="PPV Prediction", layout="centered")
logo = Image.open("logo.png")
st.image(logo, width=80)
st.title("💥 Peak Particle Velocity Prediction")
st.markdown("#### XGBoost-Based Blasting Evaluation Model")

distance = st.number_input("📏 Distance from Blast (m)", min_value=10.0, max_value=500.0, value=100.0)
charge = st.number_input("💣 Charge per Delay (kg)", min_value=1.0, max_value=1000.0, value=250.0)
rock_type = st.selectbox("🪨 Select Rock Type", ["Limestone", "Coal"])

if st.button("🔮 Predict PPV"):
    try:
        constants = rock_constants[rock_type]
        k = constants["K"]
        beta = constants["beta"]
        rt_limestone = constants["Rock_Type_Limestone"]

        scaled_distance = distance / np.sqrt(charge)
        log_sd = np.log10(scaled_distance)

        features = np.array([[distance, charge, k, beta, scaled_distance, log_sd, rt_limestone]])
        ppv = model.predict(features)[0]

        st.success(f"📈 Predicted PPV: **{ppv:.2f} mm/s**")

        if ppv <= 5:
            st.info("🟢 Vibration Impact: Safe")
        elif ppv <= 10:
            st.warning("🟡 Vibration Impact: Moderate")
        else:
            st.error("🔴 Vibration Impact: Unsafe")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown("<center>Indian Institute of Technology (BHU) Varanasi</center>", unsafe_allow_html=True)
