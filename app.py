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
OPENWEATHERMAP_API_KEY = os.getenv("WEATHER_API_KEY", "b53fa2ee256e02d9df2dee58371d93a3")  # Replace with your API key

# JavaScript for automatic location detection
st.markdown(
    """
    <script>
    function getLocation() {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                var latitude = position.coords.latitude;
                var longitude = position.coords.longitude;
                document.getElementById("latitude").value = latitude;
                document.getElementById("longitude").value = longitude;
                document.getElementById("location-form").submit();
            }, 
            function(error) {
                console.log("Error getting location: ", error);
                alert("Please allow location access for accurate weather data.");
            }
        );
    }
    window.onload = getLocation;
    </script>
    <form id="location-form">
        <input type="hidden" id="latitude" name="latitude">
        <input type="hidden" id="longitude" name="longitude">
    </form>
    """,
    unsafe_allow_html=True
)

# Retrieve location coordinates
latitude = st.query_params.get("latitude", None)
longitude = st.query_params.get("longitude", None)

# Default values
default_values = {
    "pressure": 1015.9,
    "dewpoint": 19.9,
    "humidity": 95.0,
    "cloud": 81.0,
    "windspeed": 13.7,
    "winddirection": 40,
    "sunshine": 0.0
}

# Display City Name
st.subheader("📍 Auto-Detected Location")

if latitude and longitude:
    try:
        geolocator = Nominatim(user_agent="rainfall_app")
        location_data = geolocator.reverse((latitude, longitude), language="en")

        if location_data:
            city = location_data.raw["address"].get("city", "Unknown Location")
            country = location_data.raw["address"].get("country", "")

            st.markdown(f"### 📍 **Detected Location:** {city}, {country}")

            # Fetch weather data
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
            response = requests.get(url)

            if response.status_code == 200:
                weather_data = response.json()
                temperature = weather_data['main'].get('temp', 19.9)
                humidity = weather_data['main'].get('humidity', 95.0)

                # Calculate dew point
                dewpoint = temperature - ((100 - humidity) / 5)

                default_values.update({
                    "pressure": weather_data['main'].get('pressure', 1015.9),
                    "humidity": humidity,
                    "cloud": weather_data['clouds'].get('all', 81.0),
                    "windspeed": weather_data['wind'].get('speed', 13.7),
                    "winddirection": weather_data['wind'].get('deg', 40),
                    "dewpoint": dewpoint,
                    "sunshine": 0.0
                })

                st.write("✅ **Fetched Weather Data:**")
                st.json(default_values)
            else:
                st.error("❌ Failed to fetch weather data. Using default values.")
        else:
            st.error("❌ Unable to detect location.")
    except Exception as e:
        st.error(f"❌ Error fetching location: {e}")

# Manual Weather Input
st.subheader("✏️ Manual Weather Data Input")
pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, float(default_values["pressure"]), 0.1)
dewpoint = st.slider("Dew Point (°C)", -50.0, 50.0, float(default_values["dewpoint"]), 0.1)
humidity = st.slider("Humidity (%)", 0.0, 100.0, float(default_values["humidity"]), 0.1)
cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, float(default_values["cloud"]), 0.1)
windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, float(default_values["windspeed"]), 0.1)
winddirection = st.slider("Wind Direction (°)", 0, 360, int(default_values["winddirection"]), 1)
sunshine = st.slider("Sunshine Hours", 0.0, 24.0, float(default_values["sunshine"]), 0.1)

# Predict Rainfall
if st.button("🚀 Predict Rainfall"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]])
    prediction = model.predict(input_data)
    result = "🌧️ Rainfall Expected" if prediction[0] == 1 else "☀️ No Rainfall"
    st.subheader(f"**Prediction: {result}**")
