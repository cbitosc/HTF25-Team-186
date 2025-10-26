# performance_v2.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.title("📊 Model Performance Dashboard")
st.write("""
This section provides insights into how well the trained model performs in classifying hazardous asteroids.
""")

# ---------------------------
# 1️⃣ Load model, scaler, feature list, and test data
# ---------------------------
@st.cache_resource
def load_model_and_data():
    # Load test data
    X_test = pd.read_csv('../data/processed/X_test_scaled.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()  # ensures 1D

    # Load trained model
    with open('../models/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load feature list
    feature_list = joblib.load('../models/feature_list.pkl')

    # Load scaler (optional if needed for new raw data)
    # with open('../models/scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)

    # ---------------------------
    # Ensure columns match feature list
    # ---------------------------
    missing_cols = set(feature_list) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # add missing features with 0

    extra_cols = set(X_test.columns) - set(feature_list)
    if extra_cols:
        X_test = X_test.drop(columns=list(extra_cols))

    X_test = X_test[feature_list]  # ensure correct order

    return model, X_test, y_test

model, X_test, y_test = load_model_and_data()

# ---------------------------
# 2️⃣ Predictions
# ---------------------------
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
else:
    # fallback for models without predict_proba
    y_proba = model.decision_function(X_test)
    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

# ---------------------------
# 3️⃣ Evaluation Metrics
# ---------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.markdown("### 📋 Model Evaluation Metrics")
# Show metrics without using .style
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [acc, prec, rec, f1]
})

# Format as percentages
metrics_df["Score"] = metrics_df["Score"].apply(lambda x: f"{x:.2%}")

st.dataframe(metrics_df)


# ---------------------------
# 4️⃣ Confusion Matrix
# ---------------------------
st.markdown("### 🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ---------------------------
# 5️⃣ ROC Curve
# ---------------------------
st.markdown("### 🧭 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend()
st.pyplot(fig_roc)

