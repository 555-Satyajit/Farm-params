import requests
import numpy as np
import json
from pathlib import Path
import joblib

class ModelClient:
    def __init__(self, client_id, server_url="http://localhost:5000"):
        self.client_id = client_id
        self.server_url = server_url
        self.model_path = None
        
    def fetch_model(self):
        """Fetch model from server"""
        response = requests.get(f"{self.server_url}/get_model/{self.client_id}")
        result = response.json()
        
        if result['status'] == 'success':
            self.model_path = result['model_path']
            print(f"Model successfully downloaded to {self.model_path}")
            return True
        else:
            print(f"Error fetching model: {result.get('error')}")
            return False
            
    def verify_model(self):
        """Verify model integrity"""
        response = requests.get(f"{self.server_url}/verify_model/{self.client_id}")
        result = response.json()
        
        if result.get('verified'):
            print("Model verification successful!")
        else:
            print(f"Model verification failed: {result.get('error')}")
        
        return result
    
    def predict(self, features):
        """Make predictions using the model"""
        data = {
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'client_id': self.client_id
        }
        
        response = requests.post(f"{self.server_url}/predict", json=data)
        result = response.json()
        
        if result['status'] == 'success':
            return np.array(result['prediction'])
        else:
            raise Exception(f"Prediction failed: {result.get('error')}")

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ModelClient(client_id=1)
    
    # Fetch and verify model
    client.fetch_model()
    client.verify_model()
    
    # Make predictions
    sample_features = np.array([1, 2, 3, 4, 5,6,7,8,9])  # Adjust based on your feature dimensions
    prediction = client.predict(sample_features)
    print(f"Prediction: {prediction}")