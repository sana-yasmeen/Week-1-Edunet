import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="EcoTrack – Carbon Footprint Predictor", layout="centered")

st.title("🌱 EcoTrack — Carbon Footprint Predictor")
st.write("Enter the values below to predict your carbon emission.")

# ----------------------
# Load Model
# ----------------------
MODEL_PATH = r"C:\Users\sanay\PycharmProjects\EcoTrack\week2\eco_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at: {MODEL_PATH}")
else:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")

# ----------------------
# Categorical Mapping (same as model.py)
# ----------------------
mapping = {
    "never": 0, "rarely": 1, "often": 2, "frequently": 3,

    "small": 0, "medium": 1, "large": 2,

    "yes": 1, "no": 0,

    "gas": 0, "electric": 1, "wood": 2
}

# Custom mappings for new columns
body_map = {"slim": 0, "average": 1, "fat": 2}
sex_map = {"male": 0, "female": 1}
diet_map = {"veg": 0, "non-veg": 1, "vegan": 2}
shower_map = {"short": 0, "medium": 1, "long": 2}
heating_map = {"electric": 0, "gas": 1, "solar": 2}
transport_map = {"public": 0, "private": 1}
vehicle_map = {"none": 0, "bike": 1, "car": 2}
social_map = {"low": 0, "medium": 1, "high": 2}

st.header("📌 Enter Your Details")

# -------------- ALL INPUTS (Matching CSV Columns) ----------------
body = st.selectbox("Body Type:", list(body_map.keys()))
sex = st.selectbox("Sex:", list(sex_map.keys()))
diet = st.selectbox("Diet:", list(diet_map.keys()))
shower = st.selectbox("How Often Shower:", list(shower_map.keys()))
heating = st.selectbox("Heating Energy Source:", list(heating_map.keys()))
transport = st.selectbox("Transport:", list(transport_map.keys()))
vehicle = st.selectbox("Vehicle Type:", list(vehicle_map.keys()))
social = st.selectbox("Social Activity:", list(social_map.keys()))

grocery_bill = st.number_input("Monthly Grocery Bill (₹):", min_value=0.0)
air_freq = st.selectbox("Frequency of Traveling by Air:", ["never", "rarely", "often", "frequently"])
vehicle_km = st.number_input("Vehicle Monthly Distance (km):", min_value=0.0)
waste_bag_size = st.selectbox("Waste Bag Size:", ["small", "medium", "large"])
waste_bags = st.number_input("Waste Bag Weekly Count:", min_value=0)

tv_hours = st.number_input("Daily TV/PC Hours:", min_value=0.0)
new_clothes = st.number_input("New Clothes Bought Monthly:", min_value=0)
internet_hours = st.number_input("Daily Internet Hours:", min_value=0.0)

energy = st.number_input("Home Energy Efficiency (1–10):", min_value=1, max_value=10)
recycling = st.selectbox("Recycling:", ["yes", "no"])
cooking = st.selectbox("Cooking With:", ["gas", "electric", "wood"])

# ----------------- Prediction ------------------
if st.button("Predict My Carbon Emission"):
    sample = pd.DataFrame([{
        "Body Type": body_map[body],
        "Sex": sex_map[sex],
        "Diet": diet_map[diet],
        "How Often Shower": shower_map[shower],
        "Heating Energy Source": heating_map[heating],
        "Transport": transport_map[transport],
        "Vehicle Type": vehicle_map[vehicle],
        "Social Activity": social_map[social],
        "Monthly Grocery Bill": grocery_bill,
        "Frequency of Traveling by Air": mapping[air_freq],
        "Vehicle Monthly Distance Km": vehicle_km,
        "Waste Bag Size": mapping[waste_bag_size],
        "Waste Bag Weekly Count": waste_bags,
        "How Long TV PC Daily Hour": tv_hours,
        "How Many New Clothes Monthly": new_clothes,
        "How Long Internet Daily Hour": internet_hours,
        "Energy efficiency": energy,
        "Recycling": mapping[recycling],
        "Cooking_With": mapping[cooking]
    }])

    output = model.predict(sample)[0]
    st.success(f"🌍 Estimated Carbon Emission: *{output:.2f} units*")
import seaborn as sns
import matplotlib.pyplot as plt

st.header("📊 Data Visualization")

# ------------------------
# Load Dataset for Charts
# ------------------------
DATA_PATH = r"C:\Users\sanay\PycharmProjects\EcoTrack\week2\Carbon Emission.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    # Convert categorical columns to numeric for heatmap
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    st.subheader("🔹 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🔹 Carbon Emission Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["CarbonEmission"], kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("🔹 Feature Importance Chart")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    importance = pd.Series(model.coef_, index=model.feature_names_in_)
    importance.sort_values().plot(kind="barh", ax=ax3)
    st.pyplot(fig3)

else:
    st.error(f"Dataset not found at {DATA_PATH}")