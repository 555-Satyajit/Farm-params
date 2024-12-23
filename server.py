from flask import Flask, jsonify, request
import joblib
import numpy as np
from datetime import datetime
import json
import hashlib
from pathlib import Path

class ModelDistributor:
    def __init__(self, global_model_path='global_model.pkl', version=None):
        self.global_model_path = global_model_path
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.distribution_dir = Path('distributed_models')
        self.distribution_dir.mkdir(exist_ok=True)
        
    def calculate_model_hash(self, model_data):
        """Calculate SHA-256 hash of model data for integrity verification"""
        return hashlib.sha256(str(model_data).encode()).hexdigest()
    
    def create_model_metadata(self, client_id, model_hash):
        """Create metadata for the distributed model"""
        return {
            'client_id': client_id,
            'distribution_date': datetime.now().isoformat(),
            'model_version': self.version,
            'model_hash': model_hash,
            'original_model': self.global_model_path
        }
    
    def distribute_to_client(self, client_id):
        """Distribute global model to a specific client"""
        try:
            # Load global model
            global_coef, global_intercept = joblib.load(self.global_model_path)
            
            # Create client-specific model directory
            client_dir = self.distribution_dir / f'client_{client_id}'
            client_dir.mkdir(exist_ok=True)
            
            # Create model file name with version
            model_filename = f'global_model_v{self.version}.pkl'
            model_path = client_dir / model_filename
            
            # Save model for client
            model_data = (global_coef, global_intercept)
            joblib.dump(model_data, model_path)
            
            # Calculate model hash
            model_hash = self.calculate_model_hash(model_data)
            
            # Create and save metadata
            metadata = self.create_model_metadata(client_id, model_hash)
            metadata_path = client_dir / f'metadata_v{self.version}.json'
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return {
                'status': 'success',
                'client_id': client_id,
                'model_path': str(model_path),
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'client_id': client_id,
                'error': str(e)
            }
    
    def verify_distribution(self, client_id):
        """Verify the integrity of a distributed model"""
        try:
            client_dir = self.distribution_dir / f'client_{client_id}'
            model_path = client_dir / f'global_model_v{self.version}.pkl'
            metadata_path = client_dir / f'metadata_v{self.version}.json'
            
            # Load distributed model and calculate hash
            distributed_model = joblib.load(model_path)
            current_hash = self.calculate_model_hash(distributed_model)
            
            # Load metadata and get stored hash
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            stored_hash = metadata['model_hash']
            
            # Compare hashes
            return {
                'client_id': client_id,
                'verified': current_hash == stored_hash,
                'model_version': metadata['model_version'],
                'distribution_date': metadata['distribution_date']
            }
            
        except Exception as e:
            return {
                'client_id': client_id,
                'verified': False,
                'error': str(e)
            }

# Initialize Flask app
app = Flask(__name__)
distributor = ModelDistributor(global_model_path='global_model.pkl')

@app.route('/get_model/<int:client_id>', methods=['GET'])
def get_model(client_id):
    """Endpoint to distribute model to a specific client"""
    result = distributor.distribute_to_client(client_id)
    return jsonify(result)

@app.route('/verify_model/<int:client_id>', methods=['GET'])
def verify_model(client_id):
    """Endpoint to verify model integrity"""
    result = distributor.verify_distribution(client_id)
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions"""
    try:
        # Get data from request
        data = request.json
        features = np.array(data['features'])
        client_id = data['client_id']
        
        # Load client's model
        client_dir = Path('distributed_models') / f'client_{client_id}'
        model_path = next(client_dir.glob('global_model_*.pkl'))  # Get latest version
        model_coef, model_intercept = joblib.load(model_path)
        
        # Make prediction
        prediction = np.dot(features, model_coef) + model_intercept
        
        return jsonify({
            'status': 'success',
            'prediction': prediction.tolist(),
            'client_id': client_id
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    print("Starting the model distribution server...")
    print("Server will be running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)