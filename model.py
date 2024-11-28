import numpy as np

class Neural_Network:
    def __init__(self, weights1, weights2, weights3, weights4, biase):
        self.input = 0
        self.actual_output = 0
        self.weights_1 = weights1
        self.weights_2 = weights2
        self.weights_3 = weights3
        self.weights_4 = weights4
        self.biase = biase
        self.output_1 = 0
        self.output_2 = 0
        self.output_3 = 0
        self.predicted_output = 0
        

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def set_input(self,actual_input,actual_output):
        self.input = actual_input
        self.actual_output =actual_output
        
    def infer(self,infer_input):
        self.input=infer_input
               
    def forward(self):
        self.output_1 = self.layer_1()
        self.output_2 = self.layer_2()
        self.output_3 = self.layer_3()
        self.predicted_output = self.output_layer()
        return self.predicted_output

    def backward(self):
        output_error = self.actual_output - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        hidden_error_3 = np.dot(output_delta, self.weights_4.T)
        hidden_delta_3 = hidden_error_3 * self.sigmoid_derivative(self.output_3)

        hidden_error_2 = np.dot(hidden_delta_3, self.weights_3.T)
        hidden_delta_2 = hidden_error_2 * self.sigmoid_derivative(self.output_2)

        hidden_error_1 = np.dot(hidden_delta_2, self.weights_2.T)
        hidden_delta_1 = hidden_error_1 * self.sigmoid_derivative(self.output_1)

        self.weights_4 += np.dot(self.output_3.T, output_delta) * 0.01
        self.weights_3 += np.dot(self.output_2.T, hidden_delta_3) * 0.01
        self.weights_2 += np.dot(self.output_1.T, hidden_delta_2) * 0.01
        self.weights_1 += np.dot(self.input.T, hidden_delta_1) * 0.01

    def layer_1(self):
        return self.sig(np.dot(self.input, self.weights_1) + self.biase)

    def layer_2(self):
        return self.sig(np.dot(self.output_1, self.weights_2) + self.biase)

    def layer_3(self):
        return self.sig(np.dot(self.output_2, self.weights_3) + self.biase)

    def output_layer(self):
        return self.sig(np.dot(self.output_3, self.weights_4) + self.biase)
    
    def binary_cross_entropy(y_true, y_pred):
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return bce 