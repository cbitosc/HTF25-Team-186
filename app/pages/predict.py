# predict_v3.py

import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

st.title("🔮 Asteroid Hazard Prediction")
st.write("Predict whether an asteroid is **Hazardous** or **Non-Hazardous** using the trained model.")

# ================= LOAD MODEL, SCALER & FEATURE LIST =================
@st.cache_resource
def load_model():
    with open('../models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    feature_list = joblib.load('../models/feature_list.pkl')
    return model, scaler, feature_list

model, scaler, feature_list = load_model()

# ================= VELOCITY MAPPING =================
velocity_mapping = {
    'Very Slow': 5.0,
    'Slow': 10.0,
    'Medium': 20.0,
    'Fast': 35.0,
    'Very Fast': 50.0
}

def safe_float_velocity(vel):
    """Convert velocity to float or map categorical values."""
    try:
        return float(vel)
    except ValueError:
        return velocity_mapping.get(vel, 20.0)  # default median 20 km/s

# ================= FEATURE ENGINEERING =================
def compute_features(velocity, distance):
    """Compute derived features used during training."""
    velocity = safe_float_velocity(velocity)
    distance = float(distance)

    inv_distance = 1 / (distance + 1e-9)
    log_distance = np.log1p(distance)
    log_velocity = np.log1p(velocity)

    features = {
        'Relative Velocity km per sec': velocity,
        'Miss Dist.(kilometers)': distance,
        'inv_distance': inv_distance,
        'log_distance': log_distance,
        'log_velocity': log_velocity,
    }

    # Engineered features used in training
    features['velocity_distance_ratio'] = velocity / (distance + 1e-9)
    features['log_vel_by_log_dist'] = log_velocity / (log_distance + 1e-9)

    return features

# ================= PREDICTION TYPE =================
option = st.radio("Choose Prediction Type", ["Single Prediction", "Batch Prediction"])

# ================= SINGLE PREDICTION =================
if option == "Single Prediction":
    st.subheader("🧍 Single Asteroid Prediction")

    velocity = st.text_input("Relative Velocity (km/s) or category", value="20.0")
    distance = st.number_input("Miss Distance (km)", min_value=1.0, step=1000.0, value=1_000_000.0)

    if st.button("Predict"):
        features_dict = compute_features(velocity, distance)
        input_df = pd.DataFrame([features_dict])

        # Ensure all feature_list columns exist
        for col in feature_list:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_list]

        # Scale and predict
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Potentially Hazardous (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Non-Hazardous (Confidence: {1-proba:.2%})")

# ================= BATCH PREDICTION =================
elif option == "Batch Prediction":
    st.subheader("🧾 Batch Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict Batch"):
            feature_rows = []
            for _, row in df.iterrows():
                feature_rows.append(
                    compute_features(
                        row['Relative Velocity km per sec'],
                        row['Miss Dist.(kilometers)']
                    )
                )
            feature_df = pd.DataFrame(feature_rows)

            # Ensure all feature_list columns exist
            for col in feature_list:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[feature_list]

            # Scale and predict
            scaled = scaler.transform(feature_df)
            preds = model.predict(scaled)
            probas = model.predict_proba(scaled)[:, 1]

            df['Predicted_Hazardous'] = preds
            df['Hazardous_Label'] = df['Predicted_Hazardous'].apply(lambda x: "Hazardous" if x == 1 else "Non-Hazardous")
            df['Confidence'] = probas

            st.write("### ✅ Prediction Results")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
