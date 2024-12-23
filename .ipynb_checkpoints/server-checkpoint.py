import flwr as fl
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load the initial global model
global_model = LinearRegression()
global_coef, global_intercept = joblib.load('global_model.pkl')
global_model.coef_, global_model.intercept_ = global_coef, global_intercept

# Define a function to evaluate the global model
def get_eval_fn(model):
    def evaluate(weights):
        model.coef_, model.intercept_ = weights[:-1], weights[-1]
        # Dummy data for evaluation (use a real validation set in practice)
        X_val = np.random.rand(10, 7)  # Adjust to your feature size
        y_val = np.random.rand(10)
        loss = np.mean((model.predict(X_val) - y_val) ** 2)
        return loss, {"mse": loss}
    return evaluate

# Start the Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config={"num_rounds": 3},
    strategy=fl.server.strategy.FedAvg(
        eval_fn=get_eval_fn(global_model)
    )
)
