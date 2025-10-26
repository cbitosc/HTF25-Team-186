# 01_EDA_Preprocessing_clean_fixed_v2.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import joblib

# ---------------------------
# 1️⃣ Load Data
# ---------------------------
df = pd.read_csv('../data/asteroid_data.csv')
print("✅ Columns loaded:", df.columns.tolist())

# ---------------------------
# 2️⃣ Preprocessing
# ---------------------------
numeric_cols = [
    'Relative Velocity km per sec',
    'Miss Dist.(kilometers)',
    'Jupiter Tisserand Invariant',
    'Semi Major Axis',
    'Mean Motion',
    'Orbit Uncertainity'
]

# Convert numeric columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Handle categorical columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert target column to binary
df['Hazardous'] = df['Hazardous'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# ---------------------------
# 3️⃣ Feature Engineering
# ---------------------------

# Derived distance and velocity features
df['Distance_Ratio'] = df['Miss Dist.(kilometers)'] / df['Miss Dist.(kilometers)'].max()
df['inv_distance'] = 1 / (df['Miss Dist.(kilometers)'] + 1e-9)
df['log_distance'] = np.log1p(df['Miss Dist.(kilometers)'])
df['log_velocity'] = np.log1p(df['Relative Velocity km per sec'])

# Features required by the model
df['log_vel_by_log_dist'] = df['log_velocity'] / (df['log_distance'] + 1e-9)
df['velocity_distance_ratio'] = df['Relative Velocity km per sec'] / (df['Miss Dist.(kilometers)'] + 1e-9)

# Velocity category
try:
    df['Velocity_Category'] = pd.qcut(
        df['Relative Velocity km per sec'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop'
    )
except ValueError:
    median_val = df['Relative Velocity km per sec'].median()
    df['Velocity_Category'] = df['Relative Velocity km per sec'].apply(
        lambda x: 'High' if x > median_val else 'Low'
    )

# One-hot encoding
df = pd.get_dummies(df, columns=['Velocity_Category'], drop_first=True)

# ---------------------------
# 4️⃣ Prepare features & target
# ---------------------------
drop_cols = ['Hazardous', 'Name', 'Epoch Date Close Approach']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Hazardous']

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure numeric

# ---------------------------
# 5️⃣ Handle class imbalance
# ---------------------------
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# ---------------------------
# 6️⃣ Split data
# ---------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ---------------------------
# 7️⃣ Scale features
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 8️⃣ Save processed data & models
# ---------------------------
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('../data/processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_val_scaled, columns=X_val.columns).to_csv('../data/processed/X_val_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('../data/processed/X_test_scaled.csv', index=False)

y_train.to_csv('../data/processed/y_train.csv', index=False)
y_val.to_csv('../data/processed/y_val.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

# Save scaler
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature list
feature_list = X_train.columns.tolist()
joblib.dump(feature_list, '../models/feature_list.pkl')

print("✅ Preprocessing complete. Scaler & feature list saved successfully.")
print("📊 Total features used:", len(feature_list))
