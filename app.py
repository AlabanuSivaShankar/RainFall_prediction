import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
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

# Reverse geocode function to get location name from coordinates
def get_location_name(lat, lon):
    geolocator = Nominatim(user_agent="rainfall_prediction")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        return location.address if location else "Unknown Location"
    except GeocoderTimedOut:
        return "Location fetch timed out"

# Streamlit UI
st.title("🌧️ Rainfall Prediction App")
st.write("Enter the weather conditions below to predict whether it will rain or not.")

# Default weather parameters
default_values = {
    "pressure": 1015.9,
    "dewpoint": 19.9,
    "humidity": 95.0,
    "cloud": 81.0,
    "windspeed": 13.7,
    "winddirection": 40,
    "sunshine": 0.0
}

# Fetch location-based weather data
st.subheader("📍 Location-Based Weather Data")

# Check for browser location permission
query_params = st.query_params
latitude = query_params.get("latitude")
longitude = query_params.get("longitude")

if latitude and longitude:
    try:
        lat, lon = float(latitude[0]), float(longitude[0])
        location_name = get_location_name(lat, lon)  # Get city name

        st.write(f"✅ **Detected Location:** {location_name} (Latitude: {lat}, Longitude: {lon})")

        # Fetch real-time weather data
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            weather_data = response.json()
            temperature = weather_data['main'].get('temp', 19.9)
            humidity = weather_data['main'].get('humidity', 95.0)

            # Calculate dew point (approximation)
            dewpoint = temperature - ((100 - humidity) / 5)

            default_values.update({
                "pressure": weather_data['main'].get('pressure', 1015.9),
                "humidity": humidity,
                "cloud": weather_data['clouds'].get('all', 81.0),
                "windspeed": weather_data['wind'].get('speed', 13.7),
                "winddirection": weather_data['wind'].get('deg', 40),
                "dewpoint": dewpoint,
                "sunshine": 0.0  # OpenWeatherMap does not provide sunshine data
            })

            st.write("✅ **Fetched Weather Data:**")
            st.json(default_values)
        else:
            st.error("❌ Failed to fetch weather data. Using default values.")
    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")
else:
    st.warning("⚠️ Please allow location access in your browser.")

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
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], 
                              columns=["pressure", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"])
    prediction = model.predict(input_data)
    result = "🌧️ Rainfall Expected" if prediction[0] == 1 else "☀️ No Rainfall"
    st.subheader(f"**Prediction: {result}**")
