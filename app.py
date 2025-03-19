import os
import pickle
import requests
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define model filename
MODEL_FILENAME = "rainfall_model.pkl"

# Load dataset function
@st.cache_data
def load_data():
    data = pd.read_csv("rainfall.csv")
    data["rainfall"] = data["rainfall"].str.lower().map({"yes": 1, "no": 0})
    return data

# Load dataset
data = load_data()

# Train and save model
if not os.path.exists(MODEL_FILENAME):
    feature_names = ["pressure", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
    X = data[feature_names]
    y = data["rainfall"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save trained model
    with open(MODEL_FILENAME, "wb") as file:
        pickle.dump(model, file)
else:
    with open(MODEL_FILENAME, "rb") as file:
        model = pickle.load(file)

# Streamlit UI
st.title("Rainfall Prediction System")

city = st.text_input("Enter City Name:")

if st.button("Get Weather & Predict Rainfall"):
    if city:
        API_KEY = "b53fa2ee256e02d9df2dee58371d93a3"  # Replace with actual OpenWeatherMap API key
        URL = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(URL)
        
        if response.status_code == 200:
            weather_data = response.json()
            if "main" in weather_data:
                pressure = weather_data["main"].get("pressure", 1013)
                humidity = weather_data["main"].get("humidity", 50)
                windspeed = weather_data["wind"].get("speed", 5.0)
                dewpoint = humidity * 0.1  # Simplified estimation
                cloud = weather_data.get("clouds", {}).get("all", 50)
                sunshine = 100 - cloud
                winddirection = weather_data["wind"].get("deg", 180)
                
                # Create input DataFrame
                feature_names = ["pressure", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
                input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]],
                                          columns=feature_names)
                
                # Ensure correct feature order
                input_data = input_data[feature_names]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                result = "Yes, it will rain." if prediction == 1 else "No, it will not rain."
                st.success(f"Prediction: {result}")
            else:
                st.error("Unexpected API response. Please try again.")
        else:
            st.error("Failed to fetch weather data. Check city name and API key.")
    else:
        st.warning("Please enter a city name.")
