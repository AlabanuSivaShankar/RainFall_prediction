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

# API Key for OpenWeatherMap
API_KEY = os.getenv("WEATHER_API_KEY", "b53fa2ee256e02d9df2dee58371d93a3")  # Ensure this is set securely

# Constants
DATA_PATH = "https://raw.githubusercontent.com/AlabanuSivaShankar/RainFall_prediction/main/Rainfall.csv"
MODEL_PATH = "rainfall_prediction_model.pkl"

@st.cache_data
def load_data():
    """Load dataset from GitHub."""
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def preprocess_data(data):
    """Preprocess dataset."""
    data.columns = data.columns.str.strip()
    data.drop(columns=["day"], errors='ignore', inplace=True)
    data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
    data["windspeed"].fillna(data["windspeed"].median(), inplace=True)
    data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
    data.drop(columns=['maxtemp', 'temparature', 'mintemp'], errors='ignore', inplace=True)
    return data

def train_model(data):
    """Train a RandomForest model."""
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

# Default Weather Values
default_values = {
    "pressure": 1015.9,
    "dewpoint": 19.9,
    "humidity": 95.0,
    "cloud": 81.0,
    "windspeed": 13.7,
    "winddirection": 40,
    "sunshine": 0.0
}

# Location-Based Weather Data
st.subheader("Location-Based Weather Data")
location = st.text_input("Enter Location (City, Country):")

if location:
    try:
        geolocator = Nominatim(user_agent="rainfall_app")
        location_data = geolocator.geocode(location)

        if location_data:
            lat, lon = location_data.latitude, location_data.longitude
            st.write(f"📍 Latitude: {lat}, Longitude: {lon}")

            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            response = requests.get(url)

            if response.status_code == 200:
                weather_data = response.json()
                default_values["pressure"] = weather_data['main'].get('pressure', 1013)
                default_values["humidity"] = weather_data['main'].get('humidity', 50)
                default_values["cloud"] = weather_data['clouds'].get('all', 50)
                default_values["windspeed"] = weather_data['wind'].get('speed', 5.0)
                default_values["winddirection"] = weather_data['wind'].get('deg', 180)
                
                temp = weather_data['main'].get('temp', 25)
                default_values["dewpoint"] = temp - ((100 - default_values["humidity"]) / 5)

                st.write(f"✅ Fetched Weather Data:")
                st.write(f"- **Pressure:** {default_values['pressure']} hPa")
                st.write(f"- **Dew Point:** {default_values['dewpoint']:.2f} °C")
                st.write(f"- **Humidity:** {default_values['humidity']}%")
                st.write(f"- **Cloud Cover:** {default_values['cloud']}%")
                st.write(f"- **Wind Speed:** {default_values['windspeed']} km/h")
                st.write(f"- **Wind Direction:** {default_values['winddirection']}°")

            else:
                st.error("❌ Failed to fetch weather data. Please check API key or try again later.")
        else:
            st.error("❌ Location not found. Please enter a valid location.")
    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")

# Manual Input Fields
st.subheader("Manual Weather Data Input")
pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, default_values["pressure"], 0.1)
dewpoint = st.slider("Dew Point (°C)", -50.0, 50.0, default_values["dewpoint"], 0.1)
humidity = st.slider("Humidity (%)", 0.0, 100.0, default_values["humidity"], 0.1)
cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, default_values["cloud"], 0.1)
windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, default_values["windspeed"], 0.1)
winddirection = st.slider("Wind Direction (°)", 0, 360, default_values["winddirection"], 1)
sunshine = st.slider("Sunshine Hours", 0.0, 24.0, default_values["sunshine"], 0.1)

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
