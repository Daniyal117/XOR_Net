import numpy as np
import pandas as pd
from model import Neural_Network
from utils import random_initialization, save_weights

def main():
    # Load dataset
    xor_data = pd.read_csv('Xor_Dataset.csv')
    
    # Split into training and testing datasets
    ratio = 0.75
    total_rows = xor_data.shape[0]
    train_size = int(total_rows * ratio)
    train = xor_data[0:train_size]
    test = xor_data[train_size:]

    train_input_array = train[["X", "Y"]].to_numpy()
    train_actual_output = train[["Z"]].to_numpy() 

    # Initialize weights and biases
    weights1, weights2, weights3, weights4, biase = random_initialization()

    # Initialize the feedforward object
    model = Neural_Network(weights1, weights2, weights3, weights4, biase)
    model.set_input(train_input_array, train_actual_output)
    
    # Training loop
    for epoch in range(1000):  # Number of epochs
        predicted_output = model.forward()
        model.backward()
        
    binary_predictions = (predicted_output > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == train_actual_output) * 100
    print(f"Train accuracy: {accuracy:.4f}")

        # Save the weights after training
    weights = (model.weights_1, model.weights_2, model.weights_3, model.weights_4)
    save_weights(weights, 'weights/best_weights.pkl')

    # Test the model
    test_input = test[["X", "Y"]].to_numpy()
    test_actual_output = test[["Z"]].to_numpy()
    model.set_input(test_input, test_actual_output)
    predicted_test_output = model.forward()
    test_loss = Neural_Network.binary_cross_entropy(test_actual_output, predicted_test_output)
    
    binary_predictions = (predicted_test_output > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == test_actual_output) * 100
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
