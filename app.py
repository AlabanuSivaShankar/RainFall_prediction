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
file_path = "https://raw.githubusercontent.com/AlabanuSivaShankar/RainFall_prediction/main/Rainfall.csv"

@st.cache_data
def load_data():
    return pd.read_csv(file_path)

data = load_data()

# Preprocess the data
data.columns = data.columns.str.strip()  # Strip whitespace from column names
data.drop(columns=["day"], errors='ignore', inplace=True)  # Drop unnecessary column

# Convert categorical rainfall column to numerical
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Handle missing values
data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
data["windspeed"].fillna(data["windspeed"].median(), inplace=True)

# Display dataset overview
st.title("Rainfall Prediction System")
st.subheader("Dataset Overview")
st.write(data.head())

# Display box plots in Streamlit
st.subheader("📊 Feature Box Plots")

# Define numerical columns for boxplots
boxplot_columns = ["pressure", "dewpoint", "humidity", "cloud", "windspeed"]

# Create a single figure for multiple boxplots
fig, axes = plt.subplots(1, len(boxplot_columns), figsize=(20, 5))

# Generate box plots for each feature
for i, column in enumerate(boxplot_columns):
    sns.boxplot(data=data, x="rainfall", y=column, ax=axes[i])
    axes[i].set_title(f"{column} by Rainfall")

# Adjust layout and display in Streamlit
plt.tight_layout()
st.pyplot(fig)  # ✅ Box plots will now appear in Streamlit

# Feature selection
X = data.drop(columns=["rainfall"])
y = data["rainfall"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
st.subheader("🔍 Training RandomForest Model")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Model evaluation
y_pred = rf.predict(X_test)
st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Save the trained model
model_filename = "rainfall_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(rf, model_file)
st.success("Model trained and saved successfully!")
