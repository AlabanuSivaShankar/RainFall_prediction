# RainFall_prediction



🚀 Features
✅ Real-time location-based weather data using OpenWeatherMap API

✅ Manual input for weather parameters

✅ Predicts current rainfall based on trained ML model (Random Forest)

✅ Simulates rainfall probability for next 6 months

✅ Visual representation using Seaborn bar plots

✅ Automatically handles missing data and balances dataset for accurate predictions

------------------------------------------------------------------------------------------------------------------------------
📁 Project Structure

Rainfall_Prediction_App/
│
├── rainfall_prediction_app.py      # Main Streamlit application
├── rainfall_prediction_model.pkl   # Trained model (auto-generated if not exists)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation


-------------------------------------------------------------------------------------------------------------------------------

🧠 Machine Learning Model
Model: Random Forest Classifier

Training Dataset: Rainfall.csv

Preprocessing:

Handled missing values

Dropped irrelevant columns (day, maxtemp, temparature, mintemp)

Encoded target labels (yes → 1, no → 0)

Balanced the dataset using downsampling

----------------------------------------------------------------------------------------------------------------------------------

🔧 Installation
1)Clone the Repository

git clone https://github.com/AlabanuSivaShankar/Rainfall_prediction.git
cd Rainfall_prediction


2)Install Dependencies
pip install -r requirements.txt


3)Run the Streamlit App
streamlit run rainfall_prediction_app.py

--------------------------------------------------------------------------------------------------------------------------------------
🔑 API Key Setup (Optional)

This app uses the OpenWeatherMap API for real-time weather data.
You can set your own API key using environment variables:


export WEATHER_API_KEY=your_api_key_here
If not set, a default key is used (can hit rate limits on free plans).
---------------------------------------------------------------------------------------------------------------------------------------
✨ How It Works (Step-by-Step)
1. Load Dataset
Dataset is fetched from GitHub.

Uses caching to reduce re-fetching.

2. Preprocess Data
Removes unnecessary columns.

Fills missing values.

Encodes target variable (rainfall).

3. Train Model
Balances classes using downsampling.

Splits dataset into train/test sets.

Trains a RandomForestClassifier.

4. Load or Train Model
If rainfall_prediction_model.pkl exists, loads it.

Otherwise, trains model and saves it to disk.

5. Streamlit UI
User can input location.

Weather data is fetched using Geopy + OpenWeatherMap API.

Or, manually input weather parameters via sliders.

6. Prediction
Button 1: Predict rainfall today based on current weather data.

Button 2: Simulate and visualize rainfall predictions for the next 6 months.

7. Visualization
A bar chart is displayed showing monthly rainfall probabilities.

📷 Screenshot (Optional)
Include screenshots of:

Main input page

6-month prediction results

Rainfall probability bar chart
-------------------------------------------------------------------------------------------------------------------------------------------------------

🧪 Requirements
Python 3.7+

Libraries:

VScode
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
geopy
WeatherMap API
-----------------------------------------------------------------------------------------------------------------------------------------------------------
📬 Contact
Alabanu Siva Shankar
📧 Email: alabanusiva@gmail.com
📍 Location: Vijayawada, Andhra Pradesh

