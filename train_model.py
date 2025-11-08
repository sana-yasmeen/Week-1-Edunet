# train_model.py
# Week 2: Robust training script for EcoTrack
# Place this file inside the same folder as your CSV: 'Carbon Emission.csv'

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle

# -------- CONFIG ----------
CSV_FILENAME = "Carbon Emission.csv"   # ensure this file is in the same folder as this script
MODEL_DIR = "model"
REPORTS_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------- helper to normalize column names for matching ----------
def normalize_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9]', '', s)   # remove spaces and non-alphanumeric
    return s

def find_target_column(columns):
    """
    Try to find the best match for a Carbon Emissions column using flexible matching.
    Returns the exact column name from the CSV (original).
    """
    normalized = {normalize_name(c): c for c in columns}
    candidates = []
    # direct common keys to look for
    keys_to_find = ['carbonemission', 'carbonemissions', 'carbon_emission', 'carbon_emissions', 'carbon']
    for k in keys_to_find:
        if k in normalized:
            return normalized[k]
    # fallback: find any column whose normalized form contains both 'carbon' and 'emission'
    for key, orig in normalized.items():
        if 'carbon' in key and 'emiss' in key:   # 'emiss' matches emission/emissions
            return orig
    # final fallback: try any column that contains 'carbon'
    for key, orig in normalized.items():
        if 'carbon' in key:
            return orig
    return None

# -------- 1) Load CSV ----------
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print(f"❌ File not found: '{CSV_FILENAME}'. Put it in the same folder as this script.")
    raise

print("✅ Data loaded successfully.")
print("Shape:", df.shape)
print("Columns:")
for i, c in enumerate(df.columns, 1):
    print(f"  {i}. '{c}'")

# -------- 2) Detect target column robustly ----------
target_col = find_target_column(df.columns)
if target_col is None:
    print("\n❌ Could not find a suitable Carbon Emissions column automatically.")
    print("Please check the CSV header and set the variable 'target_col' manually in the script.")
    raise SystemExit

print(f"\n🔎 Target column detected: '{target_col}'")

# -------- 3) Basic cleaning ----------
# Drop rows without target
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Replace obvious column name issues: strip leading/trailing spaces in column names
df.columns = [c.strip() for c in df.columns]

# Fill numeric NaNs with median (safe default)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# Fill object columns' NaNs with mode
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
for c in obj_cols:
    if df[c].isna().any():
        try:
            df[c] = df[c].fillna(df[c].mode()[0])
        except Exception:
            df[c] = df[c].fillna('Unknown')

# -------- 4) Prepare features & encoding ----------
# We will one-hot encode categorical (object) columns except target if target is object
features_df = df.copy()

# If target is object (string), try to convert to numeric if possible
if features_df[target_col].dtype == object:
    # try converting common numeric-looking strings to float
    try:
        features_df[target_col] = pd.to_numeric(features_df[target_col].str.replace(',',''), errors='coerce')
    except Exception:
        pass

# If conversion produced NaN rows, drop them
if features_df[target_col].isna().sum() > 0:
    features_df = features_df.dropna(subset=[target_col]).reset_index(drop=True)

# Identify categorical columns to encode (exclude the target)
categorical_cols = [c for c in features_df.select_dtypes(include=['object']).columns.tolist() if c != target_col]
if categorical_cols:
    features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

# Final feature matrix X and target y
if target_col not in features_df.columns:
    print(f"❌ After processing, target column '{target_col}' not found. Aborting.")
    raise SystemExit

X = features_df.drop(columns=[target_col])
y = features_df[target_col]

print("\n✅ Prepared features and target.")
print("Feature count:", X.shape[1])

# If feature count is zero, abort
if X.shape[1] == 0:
    print("❌ No features available after preprocessing. Check dataset columns.")
    raise SystemExit

# -------- 5) Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# -------- 6) Train model (RandomForest for robustness) ----------
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("\n⏳ Training model ... (this may take a moment)")
model.fit(X_train, y_train)
print("✅ Model training finished.")

# -------- 7) Evaluate ----------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred) if 'mean_absolute_error' in globals() else None
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation metrics:")
print(f"  MSE: {mse:.4f}")
if mae is not None:
    print(f"  MAE: {mae:.4f}")
print(f"  R2:  {r2:.4f}")

# -------- 8) Save model and metadata ----------
model_path = os.path.join(MODEL_DIR, "eco_model.pkl")
meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")

joblib.dump(model, model_path)
with open(meta_path, "wb") as f:
    pickle.dump({"feature_columns": X.columns.tolist(), "target_col": target_col}, f)

print(f"\n💾 Saved model to: {model_path}")
print(f"💾 Saved metadata to: {meta_path}")

# -------- 9) Save short report ----------
report_path = os.path.join(REPORTS_DIR, "week2_report.txt")
with open(report_path, "w") as f:
    f.write("EcoTrack - Week 2 Report\n")
    f.write("=========================\n")
    f.write(f"CSV file: {CSV_FILENAME}\n")
    f.write(f"Detected target column: {target_col}\n")
    f.write(f"Rows used: {features_df.shape[0]}\n")
    f.write(f"Features used: {X.shape[1]}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")

print(f"\n✅ Report saved to: {report_path}")
print("DONE: Week 2 training complete. Push 'model/' and 'reports/' to GitHub for Week 2 submission.")
