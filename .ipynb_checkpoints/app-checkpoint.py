import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load the pretrained model (replace with your model's file path)
model = joblib.load('global_model.pkl')

# Function to make predictions using the loaded model
def predict_yield(form_data):
    # Preparing the feature array based on user inputs
    feature_array = np.array([
        [
            form_data["soilType"],  # This should be converted into numerical format
            form_data["crop"],      # This should be converted into numerical format
            form_data["rainfall"],
            form_data["temperature"],
            form_data["fertilizer"],
            form_data["irrigation"],
            form_data["weatherCondition"],
            form_data["daysToHarvest"]
        ]
    ])

    # Preprocess the feature array (you may need to apply encoding/normalization)
    # If you're using categorical variables, they should be encoded first (e.g., using LabelEncoder)

    # Prediction using the model
    prediction = model.predict(feature_array)  # Replace with your model's predict method
    return prediction[0]

# Streamlit app layout
def main():
    st.title("Agricultural Yield Predictor")

    # Input fields
    soil_types = ["Loamy", "Clay", "Sandy", "Silt"]
    crops = ["Wheat", "Rice", "Corn", "Soybean"]
    weather_conditions = ["Sunny", "Cloudy", "Rainy", "Windy"]

    form_data = {
        "soilType": st.selectbox("Soil Type", soil_types, index=0),
        "crop": st.selectbox("Crop", crops, index=0),
        "rainfall": st.number_input("Rainfall (mm)", value=800),
        "temperature": st.number_input("Temperature (°C)", value=25),
        "fertilizer": st.checkbox("Fertilizer Used", value=True),
        "irrigation": st.checkbox("Irrigation Used", value=True),
        "weatherCondition": st.selectbox("Weather Condition", weather_conditions, index=0),
        "daysToHarvest": st.number_input("Days to Harvest", value=120)
    }

    # Prediction button
    if st.button("Predict Yield"):
        # Get the prediction from the model
        prediction = predict_yield(form_data)
        st.write(f"Predicted Yield: {prediction:.2f} tons per hectare")

        # Mock data for charting (you should replace this with real prediction data)
        predictions = {
            "client1": prediction + 0.2,
            "client2": prediction + 0.1,
            "client3": prediction + 0.3,
            "client4": prediction + 0.25,
            "global": prediction
        }

        # Prepare data for chart
        chart_data = pd.DataFrame({
            "Client": ["Client 1", "Client 2", "Client 3", "Client 4", "Global"],
            "Yield": [
                predictions["client1"], 
                predictions["client2"], 
                predictions["client3"], 
                predictions["client4"], 
                predictions["global"]
            ]
        })

        # Plot bar chart using Plotly
        fig = px.bar(chart_data, x="Client", y="Yield", title="Predicted Agricultural Yield")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
