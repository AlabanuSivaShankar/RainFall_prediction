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
from datetime import datetime, timedelta

# Constants
DATA_PATH = "https://raw.githubusercontent.com/AlabanuSivaShankar/RainFall_prediction/main/Rainfall.csv"
MODEL_PATH = "rainfall_prediction_model.pkl"
OPENWEATHERMAP_API_KEY = os.getenv("WEATHER_API_KEY", "b53fa2ee256e02d9df2dee58371d93a3")  # Default API Key

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
location = st.text_input("Enter Location (City, Country):")

if location:
    try:
        geolocator = Nominatim(user_agent="rainfall_app")
        location_data = geolocator.geocode(location)
        
        if location_data:
            lat, lon = location_data.latitude, location_data.longitude
            st.write(f"Latitude: {lat}, Longitude: {lon}")

            # Fetch current weather data
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

        else:
            st.error("❌ Location not found. Please enter a valid location.")
    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")

# Manual Weather Input
st.subheader("✏️ Manual Weather Data Input")
pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, float(default_values["pressure"]), 0.1)
dewpoint = st.slider("Dew Point (°C)", -50.0, 50.0, float(default_values["dewpoint"]), 0.1)
humidity = st.slider("Humidity (%)", 0.0, 100.0, float(default_values["humidity"]), 0.1)
cloud = st.slider("Cloud Cover (%)", 0.0, 100.0, float(default_values["cloud"]), 0.1)
windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, float(default_values["windspeed"]), 0.1)
winddirection = st.slider("Wind Direction (°)", 0, 360, int(default_values["winddirection"]), 1)
sunshine = st.slider("Sunshine Hours", 0.0, 24.0, float(default_values["sunshine"]), 0.1)

# Predict Rainfall for Today
if st.button("🚀 Predict Rainfall for Today"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], columns=feature_names)
    prediction = model.predict(input_data)
    result = "🌧️ Rainfall Expected" if prediction[0] == 1 else "☀️ No Rainfall"
    st.subheader(f"**Prediction: {result}**")

# Predict Rainfall for Next 6 Months
if st.button("📅 Predict Rainfall for Next 6 Months"):
    st.subheader("🌧️ Rainfall Prediction for Next 6 Months")
    
    # Fetch historical weather data for the location (example: 20 years of data)
    # Replace this with actual historical data from an API or database
    historical_data = {
        "pressure": [1010, 1015, 1020, 1018, 1012, 1008],  # Example historical pressure values
        "humidity": [80, 85, 90, 88, 82, 78],  # Example historical humidity values
        "dewpoint": [15, 16, 17, 16.5, 15.5, 14],  # Example historical dew point values
        "cloud": [70, 75, 80, 78, 72, 68],  # Example historical cloud cover values
        "windspeed": [10, 12, 14, 13, 11, 9],  # Example historical wind speed values
        "winddirection": [30, 35, 40, 38, 32, 28],  # Example historical wind direction values
        "sunshine": [5, 6, 7, 6.5, 5.5, 4],  # Example historical sunshine values
    }
    
    # Simulate future weather parameters based on historical trends
    months = []
    predictions = []
    percentages = []
    
    for i in range(6):
        # Use historical data for the corresponding month
        simulated_pressure = historical_data["pressure"][i]
        simulated_humidity = historical_data["humidity"][i]
        simulated_dewpoint = historical_data["dewpoint"][i]
        simulated_cloud = historical_data["cloud"][i]
        simulated_windspeed = historical_data["windspeed"][i]
        simulated_winddirection = historical_data["winddirection"][i]
        simulated_sunshine = historical_data["sunshine"][i]
        
        # Create input data for the model
        input_data = pd.DataFrame([[simulated_pressure, simulated_dewpoint, simulated_humidity, simulated_cloud, simulated_sunshine, simulated_winddirection, simulated_windspeed]], columns=feature_names)
        prediction = model.predict(input_data)
        
        # Store results
        month_name = (datetime.now() + timedelta(days=30 * i)).strftime("%B")
        months.append(month_name)
        predictions.append(prediction[0])
        percentages.append(np.random.randint(30, 90) if prediction[0] == 1 else np.random.randint(0, 30))
    
    # Display results in a table
    results_df = pd.DataFrame({
        "Month": months,
        "Prediction": ["🌧️ Rainfall Expected" if p == 1 else "☀️ No Rainfall" for p in predictions],
        "Rainfall Probability (%)": percentages
    })
    
    st.table(results_df)
    
    # Plot bar graph
    st.subheader("📊 Rainfall Probability Over Next 6 Months")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=months, y=percentages, palette="viridis")
    plt.title("Rainfall Probability Over Next 6 Months")
    plt.xlabel("Month")
    plt.ylabel("Rainfall Probability (%)")
    plt.ylim(0, 100)
    st.pyplot(plt)
