import numpy as np
from model import Neural_Network
from utils import load_weights

def inference(input_data):
    # Load best weights
    weights_1, weights_2, weights_3, weights_4 = load_weights('weights/best_weights.pkl')

    # Initialize the model with the best weights
    model = Neural_Network(weights_1, weights_2, weights_3, weights_4, biase=0.5)

    model.infer(input_data)
    predicted_output = model.forward()
    
    print(f"Inference Result: {predicted_output}")

if __name__ == "__main__":
    # Take user input from the terminal
    print("Enter the input values (X and Y) separated by a space (e.g., '0 1'):")
    user_input = input().strip()

    try:
        x, y = map(float, user_input.split())
        test_input = np.array([[x, y]])  

        inference(test_input)
    except ValueError:
        print("Invalid input! Please enter two numerical values separated by a space.")
