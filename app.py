import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('global_model.pkl')

try:
    global_coef, global_intercept = load_model()
except:
    st.error("Error: Could not load model file. Please ensure 'global_model.pkl' exists in the same directory.")
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

def predict_crop_yield(input_data):
    """
    Make prediction using the loaded coefficients and intercept
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply mappings
    input_df['Soil_Type'] = input_df['Soil_Type'].map(soil_type_mapping)
    input_df['Crop'] = input_df['Crop'].map(crop_mapping)
    input_df['Weather_Condition'] = input_df['Weather_Condition'].map(weather_condition_mapping)
    input_df['Fertilizer_Used'] = input_df['Fertilizer_Used'].map({'Yes': 1, 'No': 0})
    input_df['Irrigation_Used'] = input_df['Irrigation_Used'].map({'Yes': 1, 'No': 0})
    
    # Convert to numpy array
    input_array = input_df.to_numpy().flatten()
    
    # Make prediction using dot product
    prediction = np.dot(input_array, global_coef) + global_intercept
    return float(prediction)

# Main app
st.title("🌾 Crop Yield Prediction System")
st.write("Enter the following details to predict crop yield:")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Field Conditions")
    soil_type = st.selectbox(
        "Soil Type",
        options=list(soil_type_mapping.keys())
    )
    
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=100.0
    )
    
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=25.0
    )
    
    weather = st.selectbox(
        "Weather Condition",
        options=list(weather_condition_mapping.keys())
    )

with col2:
    st.subheader("Crop Information")
    crop = st.selectbox(
        "Crop Type",
        options=list(crop_mapping.keys())
    )
    
    days_to_harvest = st.number_input(
        "Days to Harvest",
        min_value=1,
        max_value=365,
        value=90
    )
    
    fertilizer = st.radio(
        "Fertilizer Used",
        options=["Yes", "No"]
    )
    
    irrigation = st.radio(
        "Irrigation Used",
        options=["Yes", "No"]
    )

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
        
        st.success(f"Predicted Crop Yield: {prediction:.2f}")
        
        # Additional information
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
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add footnote
st.markdown("---")
st.caption("Note: This is a prediction model based on historical data. Actual yields may vary based on additional factors.")