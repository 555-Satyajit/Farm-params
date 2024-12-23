import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load client data
local_data = pd.read_csv("data/subset_1.csv")  # Change this for each client
local_labels = local_data['Yield_tons_per_hectare'].values
local_features = local_data.drop('Yield_tons_per_hectare', axis=1).values

# Initialize local model
local_model = LinearRegression()
local_model.fit(local_features, local_labels)

# Define Flower client
class CropYieldClient(fl.client.NumPyClient):
    def get_parameters(self):
        return np.append(local_model.coef_, local_model.intercept_)
    
    def fit(self, parameters, config):
        local_model.coef_, local_model.intercept_ = parameters[:-1], parameters[-1]
        local_model.fit(local_features, local_labels)
        return np.append(local_model.coef_, local_model.intercept_), len(local_features), {}
    
    def evaluate(self, parameters, config):
        local_model.coef_, local_model.intercept_ = parameters[:-1], parameters[-1]
        loss = np.mean((local_model.predict(local_features) - local_labels) ** 2)
        return loss, len(local_features), {"mse": loss}

# Start Flower client
fl.client.start_numpy_client("127.0.0.1:8080", client=CropYieldClient())
