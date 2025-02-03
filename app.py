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

# Load the dataset
DATA_PATH = "C:/Users/Durga Prasad/Desktop/Final-year project/Rainfall.csv"  # Update the file path
MODEL_PATH = "rainfall_prediction_model.pkl"

def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

def preprocess_data(data):
    data.columns = data.columns.str.strip()
    data = data.drop(columns=["day"], errors='ignore')
    data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
    data["windspeed"].fillna(data["windspeed"].median(), inplace=True)
    data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
    data.drop(columns=['maxtemp', 'temparature', 'mintemp'], errors='ignore', inplace=True)
    return data

def train_model(data):
    df_majority = data[data["rainfall"] == 1]
    df_minority = data[data["rainfall"] == 0]
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)
    X = df_downsampled.drop(columns=["rainfall"])
    y = df_downsampled["rainfall"]
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

# Input fields
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

# Define file path
file_path = "C:/Users/Durga Prasad/Desktop/Final-year project/Rainfall.csv"

# Check if file exists before loading
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the dataset into a pandas dataframe
data = pd.read_csv(file_path)
print(type(data))

# Data exploration
data.shape
data.head()
data.tail()
print("Data Info:")
data.info()
data.columns = data.columns.str.strip()
data = data.drop(columns=["day"])  # Drop unnecessary column

# Handle missing values
data["winddirection"].fillna(data["winddirection"].mode()[0], inplace=True)
data["windspeed"].fillna(data["windspeed"].median(), inplace=True)
print("Missing values:")
print(data.isnull().sum())

# Convert categorical rainfall column to numerical
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Visualizations
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Drop highly correlated columns
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])

# Balance the dataset
majority_class = data[data["rainfall"] == 1]
minority_class = data[data["rainfall"] == 0]
majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
data_balanced = pd.concat([majority_downsampled, minority_class]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X = data_balanced.drop(columns=["rainfall"])
y = data_balanced["rainfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with hyperparameter tuning
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Model evaluation
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(X_test)
print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sample prediction
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine','winddirection', 'windspeed'])
prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")

# Save model
model_data = {"model": best_rf_model, "feature_names": X.columns.tolist()}
with open("rainfall_prediction_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

# Load and test saved model
with open("rainfall_prediction_model.pkl", "rb") as file:
    loaded_model_data = pickle.load(file)

loaded_model = loaded_model_data["model"]
feature_names = loaded_model_data["feature_names"]
input_df = pd.DataFrame([input_data], columns=feature_names)
prediction = loaded_model.predict(input_df)
print("Loaded Model Prediction:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
