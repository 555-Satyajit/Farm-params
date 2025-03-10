import streamlit as st
import os
import git
import numpy as np
import pickle
import time
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List

# First, let's include the FederatedServer and FederatedClient classes
class FederatedServer:
    def __init__(self, num_rounds: int = 5, client_fraction: float = 0.8):
        self.global_model = None
        self.num_rounds = num_rounds
        self.client_fraction = client_fraction
        self.clients = []
        
    def initialize_global_model(self):
        """Initialize global model parameters"""
        self.global_model = {
            'coefficients': None,
            'intercept': None
        }
    
    def select_clients(self) -> List[int]:
        """Randomly select a fraction of clients for each round"""
        num_clients = len(self.clients)
        num_selected = max(1, int(self.client_fraction * num_clients))
        return random.sample(range(num_clients), num_selected)
    
    def aggregate_models(self, client_models: List[Dict]) -> Dict:
        """Aggregate client models using FedAvg"""
        coef_array = []
        intercept_array = []
        
        # Get the maximum coefficient size
        max_coef_size = max(coef.shape[0] for coef in 
                          [model['coefficients'] for model in client_models])
        
        for model in client_models:
            coef = model['coefficients']
            # Pad smaller arrays with zeros
            if coef.shape[0] < max_coef_size:
                coef = np.pad(coef, (0, max_coef_size - coef.shape[0]), 
                            mode='constant', constant_values=0)
            coef_array.append(coef)
            intercept_array.append(model['intercept'])
        
        return {
            'coefficients': np.mean(coef_array, axis=0),
            'intercept': np.mean(intercept_array, axis=0)
        }

class FederatedClient:
    def __init__(self, client_id: int, data_path: str):
        self.client_id = client_id
        self.data_path = data_path
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load client's local data"""
        # Loading pre-trained coefficients and intercepts
        self.coef = np.load(os.path.join(self.data_path, f'coef_client_{self.client_id}.npy'))
        self.intercept = np.load(os.path.join(self.data_path, 
                                            f'intercept_client_{self.client_id}.npy'))
    
    def train_local_model(self, global_model=None):
        """Train local model using client's data"""
        if global_model is not None and global_model['coefficients'] is not None:
            # Initialize local model with global parameters
            self.coef = global_model['coefficients']
            self.intercept = global_model['intercept']
        
        return {
            'coefficients': self.coef,
            'intercept': self.intercept
        }

# Streamlit app functions
def initialize_session_state():
    if 'round_number' not in st.session_state:
        st.session_state.round_number = 0
    if 'server' not in st.session_state:
        st.session_state.server = None
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'client_metrics' not in st.session_state:
        st.session_state.client_metrics = []
    if 'global_model' not in st.session_state:
        st.session_state.global_model = None

def clone_repository():
    repo_url = "https://github.com/555-Satyajit/Farm-params.git"
    local_dir = "Farm-params"
    
    if not os.path.exists(local_dir):
        try:
            with st.spinner("Cloning repository..."):
                git.Repo.clone_from(repo_url, local_dir)
            st.success("Repository cloned successfully!")
        except git.exc.GitCommandError as e:
            st.error(f"Error during cloning: {e}")
            return False
    return True

def train_federated_model():
    server = FederatedServer(
        num_rounds=st.session_state.num_rounds,
        client_fraction=st.session_state.client_fraction
    )
    server.initialize_global_model()
    
    # Initialize clients
    for client_id in range(1, 5):
        client = FederatedClient(client_id, "Farm-params")
        try:
            client.load_data()
            server.clients.append(client)
        except Exception as e:
            st.error(f"Error initializing client {client_id}: {e}")
            return None
            
    return server

def run_training_round(server):
    selected_clients = server.select_clients()
    client_models = []
    
    # Update progress metrics
    round_metrics = {
        'round': st.session_state.round_number + 1,
        'selected_clients': [i+1 for i in selected_clients],
        'client_coefficients': []
    }
    
    # Collect client updates
    for client_idx in selected_clients:
        client = server.clients[client_idx]
        client_model = client.train_local_model(server.global_model)
        client_models.append(client_model)
        round_metrics['client_coefficients'].append(
            np.mean(client_model['coefficients'])
        )
    
    # Aggregate models
    server.global_model = server.aggregate_models(client_models)
    round_metrics['global_coefficient'] = np.mean(server.global_model['coefficients'])
    
    st.session_state.client_metrics.append(round_metrics)
    st.session_state.round_number += 1
    
    return server

def display_training_progress():
    if st.session_state.client_metrics:
        # Create DataFrame for metrics
        df = pd.DataFrame(st.session_state.client_metrics)
        
        # Create line plot for model convergence
        fig = go.Figure()
        
        # Add lines for each client's coefficients
        for i in range(len(df['client_coefficients'].iloc[0])):
            client_coef = [round['client_coefficients'][i] 
                         for round in st.session_state.client_metrics]
            fig.add_trace(go.Scatter(
                x=df['round'],
                y=client_coef,
                mode='lines+markers',
                name=f'Client {i+1}'
            ))
        
        # Add line for global model
        fig.add_trace(go.Scatter(
            x=df['round'],
            y=df['global_coefficient'],
            mode='lines+markers',
            name='Global Model',
            line=dict(width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Model Convergence Over Training Rounds',
            xaxis_title='Round',
            yaxis_title='Average Coefficient Value',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)

def main():
    st.title("Federated Learning Dashboard")
    
    initialize_session_state()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Training Parameters")
        st.session_state.num_rounds = st.slider(
            "Number of Rounds", 
            min_value=1, 
            max_value=10, 
            value=5
        )
        st.session_state.client_fraction = st.slider(
            "Client Fraction", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.8
        )
        
        if not st.session_state.training_complete:
            if st.button("Start Training"):
                if clone_repository():
                    st.session_state.server = train_federated_model()
                    st.session_state.training_complete = False
    
    # Main content area
    if st.session_state.server and not st.session_state.training_complete:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Progress")
            progress_text = f"Round {st.session_state.round_number + 1}/{st.session_state.num_rounds}"
            progress_value = st.session_state.round_number / st.session_state.num_rounds
            progress_bar = st.progress(progress_value)
            st.text(progress_text)
        
        with col2:
            if st.button("Run Next Round"):
                st.session_state.server = run_training_round(st.session_state.server)
                
                if st.session_state.round_number >= st.session_state.num_rounds:
                    st.session_state.training_complete = True
                    # Save final global model
                    with open('globalmodel1.pkl', 'wb') as f:
                        pickle.dump(st.session_state.server.global_model, f)
                    st.success("Training complete! Model saved as globalmodel1.pkl")
        
        # Display training visualization
        display_training_progress()
        
    elif st.session_state.training_complete:
        st.success("Training complete! You can start a new training session from the sidebar.")
        display_training_progress()

if __name__ == "__main__":
    main()