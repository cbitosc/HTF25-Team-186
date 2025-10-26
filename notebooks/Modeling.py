# Modeling_Evaluation_fixed_v3.py

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ---------------------------
# 1️⃣ Load preprocessed data
# ---------------------------
print("📂 Loading preprocessed datasets...")
X_train = pd.read_csv('../data/processed/X_train_scaled.csv')
X_val = pd.read_csv('../data/processed/X_val_scaled.csv')
X_test = pd.read_csv('../data/processed/X_test_scaled.csv')

y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
y_val = pd.read_csv('../data/processed/y_val.csv').values.ravel()
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

# ---------------------------
# 2️⃣ Feature Engineering
# ---------------------------
def add_engineered_features(df):
    # Ensure no zero or negative values
    df['Relative Velocity km per sec'] = df['Relative Velocity km per sec'].clip(lower=1e-6)
    df['Miss Dist.(kilometers)'] = df['Miss Dist.(kilometers)'].clip(lower=1e-6)
    
    df['log_velocity'] = np.log1p(df['Relative Velocity km per sec'])
    df['log_distance'] = np.log1p(df['Miss Dist.(kilometers)'])
    df['inv_distance'] = 1 / df['Miss Dist.(kilometers)']
    df['velocity_distance_ratio'] = df['Relative Velocity km per sec'] / df['Miss Dist.(kilometers)']
    df['log_vel_by_log_dist'] = df['log_velocity'] / df['log_distance']
    
    # Replace any NaN or inf with 0 (just in case)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

X_train = add_engineered_features(X_train)
X_val = add_engineered_features(X_val)
X_test = add_engineered_features(X_test)

# ---------------------------
# 3️⃣ Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ---------------------------
# 4️⃣ Handle imbalance with SMOTE
# ---------------------------
smote = SMOTE(random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

# ---------------------------
# 5️⃣ Feature List
# ---------------------------
try:
    feature_list = joblib.load('../models/feature_list.pkl')
    print("✅ Loaded feature list successfully.")
except FileNotFoundError:
    feature_list = X_train_scaled.columns.tolist()
    print("⚠️ Feature list not found — using dataset columns instead.")

# Ensure all datasets have same columns
for df_name, df in zip(['X_train', 'X_val', 'X_test'], [X_train_scaled, X_val_scaled, X_test_scaled]):
    missing_cols = set(feature_list) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    extra_cols = set(df.columns) - set(feature_list)
    df.drop(columns=list(extra_cols), inplace=True)
    df = df[feature_list]

    if df_name == 'X_train':
        X_train_scaled = df
    elif df_name == 'X_val':
        X_val_scaled = df
    else:
        X_test_scaled = df

print("✅ Features aligned with feature list.")

# ---------------------------
# 6️⃣ Hyperparameter tuning
# ---------------------------
print("🔍 Performing hyperparameter tuning...")
rf = RandomForestClassifier(random_state=42, class_weight="balanced_subsample")

param_dist = {
    'n_estimators': [200, 400, 600, 800],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True, False]
}

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train_scaled, y_train)
model = search.best_estimator_
print(f"✅ Best hyperparameters found: {search.best_params_}")

# ---------------------------
# 7️⃣ Cross-validation
# ---------------------------
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"📊 Cross-validated F1 Score (5 folds): {np.mean(cv_scores):.4f}")

# ---------------------------
# 8️⃣ Train final model
# ---------------------------
print("🚀 Training final RandomForestClassifier...")
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# ---------------------------
# 9️⃣ Evaluate model
# ---------------------------
metrics_summary = []

for split_name, X, y in [('Validation', X_val_scaled, y_val), ('Test', X_test_scaled, y_test)]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    metrics_summary.append((split_name, acc, f1, roc))

    print(f"\n--- {split_name} Metrics ---")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC-AUC      : {roc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y, y_pred))

# ---------------------------
# 🔟 Feature importance
# ---------------------------
importances = model.feature_importances_
features = X_train_scaled.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("\n🌲 Top 15 Important Features:\n", feat_imp.head(15))

low_importance = [col for col, imp in zip(features, importances) if imp < 0.005]
print(f"⚙️ Low-importance features (consider removing): {low_importance}")

# ---------------------------
# 💾 Save model & feature list
# ---------------------------
with open('../models/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
joblib.dump(list(X_train_scaled.columns), '../models/feature_list.pkl')
print("💾 Model and feature list saved successfully.")

# ---------------------------
# 📝 Save performance summary
# ---------------------------
with open('../models/model_summary.txt', 'w') as f:
    for name, acc, f1, roc in metrics_summary:
        f.write(f"{name}: Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={roc:.4f}\n")

print("✅ Modeling and evaluation pipeline completed successfully!")
