import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import requests
from geopy.geocoders import Nominatim
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Constants
DATA_PATH = "https://raw.githubusercontent.com/AlabanuSivaShankar/RainFall_prediction/main/Rainfall.csv"
MODEL_PATH = "rainfall_prediction_model.pkl"
OPENWEATHERMAP_API_KEY = os.getenv("WEATHER_API_KEY", "b53fa2ee256e02d9df2dee58371d93a3")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Preprocess dataset
def preprocess_data(data):
    data.columns = data.columns.str.strip()
    data.drop(columns=["day"], errors='ignore', inplace=True)
    data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
    data["windspeed"].fillna(data["windspeed"].median(), inplace=True)
    data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
    data.drop(columns=['maxtemp', 'temparature', 'mintemp'], errors='ignore', inplace=True)
    return data

# Train model
def train_model(data):
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

# Fetch location-based weather data
def get_weather_by_location(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['main'].get('temp', 19.9)
        humidity = weather_data['main'].get('humidity', 95.0)
        dewpoint = temperature - ((100 - humidity) / 5)
        return {
            "pressure": weather_data['main'].get('pressure', 1015.9),
            "humidity": humidity,
            "cloud": weather_data['clouds'].get('all', 81.0),
            "windspeed": weather_data['wind'].get('speed', 13.7),
            "winddirection": weather_data['wind'].get('deg', 40),
            "dewpoint": dewpoint,
            "sunshine": 0.0
        }
    return None

# Location input
st.subheader("📍 Location-Based Weather Data")
use_device_location = st.checkbox("Use My Location")
location = ""

if use_device_location:
    location_data = st.experimental_get_query_params()
    if 'lat' in location_data and 'lon' in location_data:
        lat, lon = float(location_data['lat'][0]), float(location_data['lon'][0])
        st.write(f"🌍 **Detected Location:** Latitude {lat}, Longitude {lon}")
        default_values = get_weather_by_location(lat, lon) or {}
    else:
        st.warning("⚠️ Please allow location access in your browser.")
else:
    location = st.text_input("Enter Location (City, Country):")
    if location:
        geolocator = Nominatim(user_agent="rainfall_app")
        location_data = geolocator.geocode(location)
        if location_data:
            lat, lon = location_data.latitude, location_data.longitude
            st.write(f"🌍 **Detected Location:** Latitude {lat}, Longitude {lon}")
            default_values = get_weather_by_location(lat, lon) or {}
        else:
            st.error("❌ Location not found. Please enter a valid location.")

# Manual Weather Input
st.subheader("✏️ Manual Weather Data Input")
pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, float(default_values.get("pressure", 1015.9)), 0.1)
dewpoint = st.slider("Dew Point (°C)", -50.0, 50.0, float(default_values.get("dewpoint", 19.9)), 0.1)
humidity = st.slider("Humidity (%)", 0.0, 100.0, float(default_values.get("humidity", 95.0)), 0.1)
cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, float(default_values.get("cloud", 81.0)), 0.1)
windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, float(default_values.get("windspeed", 13.7)), 0.1)
winddirection = st.slider("Wind Direction (°)", 0, 360, int(default_values.get("winddirection", 40)), 1)
sunshine = st.slider("Sunshine Hours", 0.0, 24.0, float(default_values.get("sunshine", 0.0)), 0.1)

# Predict Rainfall
if st.button("🚀 Predict Rainfall"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], columns=feature_names)
    prediction = model.predict(input_data)
    result = "🌧️ Rainfall Expected" if prediction[0] == 1 else "☀️ No Rainfall"
    st.subheader(f"**Prediction: {result}**")
