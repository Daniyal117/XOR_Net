import numpy as np
from model import Neural_Network
from utils import load_weights

def inference(input_data: np.ndarray) -> None:
    """
    Perform inference on the input data using the trained neural network model and output 
    a binary result based on the predicted value.

    Parameters:
        input_data (np.ndarray): A 2D numpy array of shape (1, 2), where each row 
                                 represents an input example with two features (X and Y).

    Returns:
        None: This function prints the predicted output in binary format (0 or 1) based on 
              a threshold of 85. If the predicted value is greater than 85, it prints 1; 
              otherwise, it prints 0.
    """
    # Load best weights
    weights_1, weights_2, weights_3, weights_4 = load_weights('weights/best_weights.pkl')

    # Initialize the model with the best weights
    model = Neural_Network(weights_1, weights_2, weights_3, weights_4, biase=0.5)

    # Perform inference
    model.infer(input_data)
    predicted_output = model.forward()

    # Print the raw inference result
    print(f"Inference Result (raw): {predicted_output}")

    # Add condition to print 1 if output > 85, otherwise print 0
    if predicted_output > 85:
        print(1)
    else:
        print(0)

if __name__ == "__main__":
    """
    Main entry point for the script. Takes user input from the terminal, processes it,
    and performs inference using the trained neural network. After inference, the output 
    is printed as a binary value (0 or 1) based on a threshold of 85.
    """
    print("Enter the input values (X and Y) separated by a space (e.g., '0 1'):")
    user_input = input().strip()

    try:
        # Parse user input into two float values
        x, y = map(float, user_input.split())
        test_input = np.array([[x, y]])  

        # Call the inference function
        inference(test_input)
    except ValueError:
        print("Invalid input! Please enter two numerical values separated by a space.")
