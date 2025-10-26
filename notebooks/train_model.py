# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import lightgbm as lgb
# ============ CONFIG ==============
DATA_PATH = "data/dataset.csv"
MODEL_DIR = "models"
TARGET = "Hazardous"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============ LOAD DATA ============
df = pd.read_csv(DATA_PATH)
print("✅ Loaded dataset:", df.shape)
print(df.head())

# ============ FEATURE SELECTION ============
core_features = [
    'Diameter',
    'Relative Velocity (km per sec)',
    'Miss Dist. (Kilometers)'
]

extra_features = [
    'Jupiter Tisserand Invariant', 'Semi Major Axis',
    'Perihelion Arg', 'Aphelion Dist', 'MeanMotion',
    'Orbit Uncertainty', 'Orbital Period'
]

# Only keep columns that actually exist in your dataset
available_features = [f for f in core_features + extra_features if f in df.columns]

# Drop missing targets
df = df.dropna(subset=[TARGET])

# ============ FEATURE ENGINEERING ============
# Clean numeric columns
for col in available_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Derived features
df['kinetic_energy'] = 0.5 * (df['Diameter'] * 1000)**3 * (df['Relative Velocity (km per sec)'] * 1000)**2
df['inv_distance'] = 1 / (df['Miss Dist. (Kilometers)'] + 1e-9)
df['log_distance'] = np.log1p(df['Miss Dist. (Kilometers)'])
df['log_velocity'] = np.log1p(df['Relative Velocity (km per sec)'])
df['log_diameter'] = np.log1p(df['Diameter'])

# Final feature set
features = available_features + ['kinetic_energy', 'inv_distance', 'log_distance', 'log_velocity', 'log_diameter']
print("📊 Using features:", features)

# ============ SPLIT DATA ============
X = df[features]
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ============ SCALE ============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ============ MODELS ============
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "LightGBM": lgb.LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31
    )

#    "XGBoost": xgb.XGBClassifier(
#        use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1
#    )
}

results = {}
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)
    print(f"{name}: ACC={acc:.3f}, ROC-AUC={roc:.3f}")
    print(classification_report(y_test, preds))

    results[name] = {"model": model, "accuracy": acc, "roc": roc}

# ============ SAVE BEST MODEL ============
best_model_name = max(results, key=lambda n: results[n]['roc'])
best_model = results[best_model_name]['model']
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(features, os.path.join(MODEL_DIR, "feature_list.pkl"))

# Save summary
summary_path = os.path.join(MODEL_DIR, "model_summary.txt")
with open(summary_path, "w") as f:
    for name, v in results.items():
        f.write(f"{name}: ACC={v['accuracy']:.3f}, ROC={v['roc']:.3f}\n")
print(f"\n✅ Saved best model ({best_model_name}) and scaler in /models/")
