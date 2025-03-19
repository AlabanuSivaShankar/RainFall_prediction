import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
from geopy.geocoders import Nominatim
import requests

API_KEY = os.getenv("WEATHER_API_KEY")

# Constants
DATA_PATH = "https://raw.githubusercontent.com/AlabanuSivaShankar/RainFall_prediction/main/Rainfall.csv"
MODEL_PATH = "rainfall_prediction_model.pkl"
OPENWEATHERMAP_API_KEY = "b53fa2ee256e02d9df2dee58371d93a3"  # Replace with your actual API key

@st.cache_data
def load_data():
    """Load dataset from GitHub"""
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def preprocess_data(data):
    """Preprocess dataset"""
    data.columns = data.columns.str.strip()
    data.drop(columns=["day"], errors='ignore', inplace=True)
    data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
    data["windspeed"].fillna(data["windspeed"].median(), inplace=True)
    data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
    data.drop(columns=['maxtemp', 'temparature', 'mintemp'], errors='ignore', inplace=True)
    return data

def train_model(data):
    """Train a RandomForest model"""
    df_majority = data[data["rainfall"] == 1]
    df_minority = data[data["rainfall"] == 0]
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_balanced.drop(columns=["rainfall"])
    y = df_balanced["rainfall"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, X.columns.tolist()

# Load or train model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
else:
    data = load_data()
    data = preprocess_data(data)
    model, feature_names = train_model(data)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump({"model": model, "feature_names": feature_names}, file)

# Streamlit UI
st.title("🌧️ Rainfall Prediction App")
st.write("Enter the weather conditions below to predict whether it will rain or not.")

# Location-Based Weather Data
st.subheader("Location-Based Weather Data")
location = st.text_input("Enter Location (City, Country):")

pressure, dewpoint, humidity, cloud, windspeed, winddirection, sunshine = None, None, None, None, None, None, 0

if location:
    geolocator = Nominatim(user_agent="rainfall_app")
    location_data = geolocator.geocode(location)

    if location_data:
        lat, lon = location_data.latitude, location_data.longitude
        st.write(f"📍 Latitude: {lat}, Longitude: {lon}")

        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            weather_data = response.json()
            pressure = weather_data['main'].get('pressure', 1013)  # Default: 1013 hPa
            humidity = weather_data['main'].get('humidity', 50)  # Default: 50%
            cloud = weather_data['clouds'].get('all', 50)  # Default: 50%
            windspeed = weather_data['wind'].get('speed', 5.0)  # Default: 5 km/h
            winddirection = weather_data['wind'].get('deg', 180)  # Default: 180°
            
            # Calculate Approximate Dew Point using a basic formula
            temp = weather_data['main'].get('temp', 25)  # Default: 25°C
            dewpoint = temp - ((100 - humidity) / 5)  # Approximate formula

            st.write(f"✅ Fetched Weather Data:")
            st.write(f"- **Pressure:** {pressure} hPa")
            st.write(f"- **Dew Point:** {dewpoint:.2f} °C")
            st.write(f"- **Humidity:** {humidity}%")
            st.write(f"- **Cloud Cover:** {cloud}%")
            st.write(f"- **Wind Speed:** {windspeed} km/h")
            st.write(f"- **Wind Direction:** {winddirection}°")

        else:
            st.error("❌ Failed to fetch weather data. Please check API key or try again later.")
    else:
        st.error("❌ Location not found. Please enter a valid location.")

# Manual Input Fields (Backup)
st.subheader("Manual Weather Data Input")
pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, pressure, 0.1)
dewpoint = st.slider("Dew Point (°C)", -50.0, 50.0, dewpoint, 0.1)
humidity = st.slider("Humidity (%)", 0.0, 100.0, humidity, 0.1)
cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, cloud, 0.1)
windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, windspeed, 0.1)
winddirection = st.slider("Wind Direction (°)", 0, 360, winddirection, 1)
sunshine = st.slider("Sunshine Hours", 0.0, 24.0, sunshine, 0.1)

# Prediction Button
if st.button("Predict Rainfall"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], columns=feature_names)
    prediction = model.predict(input_data)
    result = "🌧️ Rainfall Expected" if prediction[0] == 1 else "☀️ No Rainfall"
    st.subheader(f"Prediction: {result}")

# Data Visualizations
data = load_data()
data = preprocess_data(data)

st.subheader("Feature Distributions")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
columns_to_plot = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
for i, column in enumerate(columns_to_plot):
    row, col = divmod(i, 3)
    sns.histplot(data[column], kde=True, ax=axes[row, col])
    axes[row, col].set_title(f"Distribution of {column}")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
