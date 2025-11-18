import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# ------------------
# Load Dataset
# ------------------
df = pd.read_csv("Carbon Emission.csv")

# ------------------
# Identify NON-NUMERIC columns
# ------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes

# ------------------
# Split Features & Target
# ------------------
X = df.drop("CarbonEmission", axis=1)
y = df["CarbonEmission"]

# ------------------
# Split Train/Test
# ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------
# Train Model
# ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------
# Save Model
# ------------------
joblib.dump(model, "eco_model.pkl")

print("Model trained and saved as eco_model.pkl")