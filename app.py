import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Load dataset
DATA_PATH = "https://raw.githubusercontent.com/AlabanuSivaShankar/RainFall_prediction/main/Rainfall.csv"
MODEL_PATH = "rainfall_prediction_model.pkl"

def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Preprocess dataset
def preprocess_data(data):
    data.columns = data.columns.str.strip()
    data = data.drop(columns=["day"], errors='ignore')
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

# User Input Fields
pressure = st.number_input("Pressure (hPa)", value=1015.9)
dewpoint = st.number_input("Dew Point (°C)", value=19.9)
humidity = st.number_input("Humidity (%)", value=95.0)
cloud = st.number_input("Cloud Cover (%)", value=81.0)
sunshine = st.number_input("Sunshine Hours", value=0.0)
winddirection = st.number_input("Wind Direction (°)", value=40.0)
windspeed = st.number_input("Wind Speed (km/h)", value=13.7)

if st.button("Predict Rainfall"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]],
                              columns=feature_names)
    prediction = model.predict(input_data)
    result = "🌧️ Rainfall Expected" if prediction[0] == 1 else "☀️ No Rainfall"
    st.subheader(f"Prediction: {result}")

# Load and process data for visualizations
data = load_data()
data = preprocess_data(data)

# Feature Distributions
st.subheader("Feature Distributions")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
columns_to_plot = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']

for i, column in enumerate(columns_to_plot):
    row, col = divmod(i, 3)
    sns.histplot(data[column], kde=True, ax=axes[row, col])
    axes[row, col].set_title(f"Distribution of {column}")

plt.tight_layout()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Box Plot Analysis
st.subheader("Feature Box Plots")
boxplot_columns = ['pressure', 'dewpoint', 'humidity', 'cloud', 'windspeed']
fig, axes = plt.subplots(1, len(boxplot_columns), figsize=(20, 5))

for i, column in enumerate(boxplot_columns):
    sns.boxplot(data=data, x="rainfall", y=column, ax=axes[i])
    axes[i].set_title(f"{column} by Rainfall")

plt.tight_layout()
st.pyplot(fig)  # Ensure boxplots appear
