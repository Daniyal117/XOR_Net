import os
import numpy as np
import pickle

def random_initialization():
    weights1 = np.random.randn(2, 3)
    weights2 = np.random.randn(3, 3)
    weights3 = np.random.randn(3, 3)
    weights4 = np.random.randn(3, 1)
    biase = 0.5
    return weights1, weights2, weights3, weights4, biase

def binary_cross_entropy(y_true, y_pred):
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce 

def save_weights(weights, file_path):
    # Ensure the directory exists before saving the weights
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save weights to a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(file_path):
    # Load weights from a pickle file
    with open(file_path, 'rb') as f:
        return pickle.load(f)
