import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
import datetime

# Set page config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# MongoDB configuration
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'yield_prediction'
COLLECTION_NAME = 'predictions'

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Load the model
@st.cache_resource
def load_model():
    with open('globalmodel1.pkl', 'rb') as f:
        model_dict = joblib.load(f)
    return model_dict['coefficients'], model_dict['intercept']

try:
    global_coef, global_intercept = load_model()
    st.write("Model loaded successfully!")
    st.write("Coefficient shape:", global_coef.shape)
except Exception as e:
    st.error(f"Error: Could not load model file. Please ensure 'globalmodel1.pkl' exists in the same directory. Error: {str(e)}")
    st.stop()

# Mappings
soil_type_mapping = {
    'Sandy': 4,
    'Clay': 1,
    'Loam': 2,
    'Silt': 5,
    'Peaty': 3,
    'Chalky': 0
}

crop_mapping = {
    'Cotton': 1,
    'Rice': 3,
    'Barley': 0,
    'Soybean': 4,
    'Wheat': 5,
    'Maize': 2
}

weather_condition_mapping = {
    'Sunny': 2,
    'Rainy': 1,
    'Cloudy': 0
}

def engineer_features(input_data):
    features = []
    features.extend([
        soil_type_mapping[input_data['Soil_Type']],
        crop_mapping[input_data['Crop']],
        float(input_data['Rainfall_mm']),
        float(input_data['Temperature_Celsius']),
        1 if input_data['Fertilizer_Used'] == 'Yes' else 0,
        1 if input_data['Irrigation_Used'] == 'Yes' else 0,
        weather_condition_mapping[input_data['Weather_Condition']],
        float(input_data['Days_to_Harvest'])
    ])
    rainfall = float(input_data['Rainfall_mm'])
    temp = float(input_data['Temperature_Celsius'])
    days = float(input_data['Days_to_Harvest'])
    features.extend([
        rainfall * temp,
        rainfall * days,
        temp * days,
        rainfall ** 2,
        temp ** 2,
        days ** 2,
        np.log1p(rainfall),
        np.log1p(temp + 21),
        np.log1p(days)
    ])
    return np.array(features, dtype=np.float64)

def predict_crop_yield(input_data):
    try:
        input_array = engineer_features(input_data)
        if input_array.shape[0] != global_coef.shape[0]:
            st.error(f"Input shape {input_array.shape} doesn't match coefficient shape {global_coef.shape}")
            st.write("Engineered features:", input_array)
            return None
        prediction = np.dot(input_array, global_coef) + global_intercept
        return float(prediction)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Main app
st.title("🌾 Crop Yield Prediction System")
st.write("Enter the following details to predict crop yield:")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Field Conditions")
    soil_type = st.selectbox("Soil Type", options=list(soil_type_mapping.keys()))
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
    weather = st.selectbox("Weather Condition", options=list(weather_condition_mapping.keys()))

with col2:
    st.subheader("Crop Information")
    crop = st.selectbox("Crop Type", options=list(crop_mapping.keys()))
    days_to_harvest = st.number_input("Days to Harvest", min_value=1, max_value=365, value=90)
    fertilizer = st.radio("Fertilizer Used", options=["Yes", "No"])
    irrigation = st.radio("Irrigation Used", options=["Yes", "No"])

# Prediction button
if st.button("Predict Yield", type="primary"):
    try:
        input_data = {
            'Soil_Type': soil_type,
            'Crop': crop,
            'Rainfall_mm': rainfall,
            'Temperature_Celsius': temperature,
            'Fertilizer_Used': fertilizer,
            'Irrigation_Used': irrigation,
            'Weather_Condition': weather,
            'Days_to_Harvest': days_to_harvest
        }
        prediction = predict_crop_yield(input_data)
        if prediction is not None:
            st.success(f"Predicted Crop Yield: {prediction:.2f}")
            st.info("Prediction Details:")
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Soil Type", soil_type)
                st.metric("Rainfall", f"{rainfall} mm")
            with col4:
                st.metric("Crop Type", crop)
                st.metric("Temperature", f"{temperature}°C")
            with col5:
                st.metric("Days to Harvest", days_to_harvest)
                st.metric("Weather", weather)

            # Save prediction to MongoDB
            prediction_data = {
                'Soil_Type': soil_type,
                'Crop': crop,
                'Rainfall_mm': rainfall,
                'Temperature_Celsius': temperature,
                'Fertilizer_Used': fertilizer,
                'Irrigation_Used': irrigation,
                'Weather_Condition': weather,
                'Days_to_Harvest': days_to_harvest,
                'Predicted_Yield': prediction,
                'Timestamp': datetime.datetime.utcnow()
            }
            collection.insert_one(prediction_data)
            st.success("Prediction logged successfully!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.caption("Note: This is a prediction model based on historical data. Actual yields may vary based on additional factors.")
