# XOR Neural Network from Scratch

## Overview
This project implements a neural network from scratch to solve the XOR problem using three hidden layers, each containing three neurons. The XOR (exclusive OR) function is a classic problem in machine learning and demonstrates the capabilities of neural networks to learn non-linear decision boundaries.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)



## Features
- **Custom Implementation**: A complete neural network built from scratch without relying on high-level libraries like TensorFlow or PyTorch.
- **Multi-layer Architecture**: Three hidden layers with three neurons each, allowing for complex function approximation.
- **Activation Functions**: Utilizes sigmoid activation functions for the hidden layers and output layer to introduce non-linearity.
- **Backpropagation**: Implements backpropagation for efficient weight updates based on error gradients.
- **XOR Dataset**: Trains specifically on the XOR dataset, showcasing the model's ability to learn non-linear relationships.

## Architecture
The architecture of the neural network consists of:
- **Input Layer**: 2 input neurons (representing the two binary inputs of the XOR function).
- **Hidden Layers**: 
  - Layer 1: 3 neurons
  - Layer 2: 3 neurons
  - Layer 3: 3 neurons
- **Output Layer**: 1 output neuron (providing the XOR result).

The model uses a feedforward approach where data flows from the input layer through hidden layers to the output layer.

## Getting Started
To get started with this repository, follow the instructions below.
### Clone the Repository
First, clone this repository to your local machine using the following command:
```
git clone https://github.com/Daniyal117/XOR_Net.git
```

### Create a Python Environment
It's recommended to use a Python virtual environment to manage dependencies. Navigate to the root directory of the repository in your terminal and run the following commands to create and activate a virtual environment:
```bash
# For Linux/MacOS
python3 -m venv env
source env/bin/activate
# For Windows
python -m venv env
.\env\Scripts\activate
```
This will create a virtual environment named `env` and activate it.
### Install Requirements
Once the virtual environment is activated, you can install the required dependencies using `pip` with the following command:
```bash
pip install -r requirments.txt
```

This will install all the dependencies listed in the `requirements.txt` file.

### To Run Script
To train and test the neural network on the XOR dataset, run the following command:
```bash
python3 main.py
```
### Inference/Prediction -->

To make predictions using the trained model, use the following command:
```bash
 python3 inference.py
```
### How It Works
Training
The model is trained using the following 

parameters:
Learning Rate: Set to 0.01, which controls how much to change the weights during training.

Epochs: Set to 10,000, indicating how many times the training algorithm will work through the entire training dataset.

Loss Function: binary cross entropy is used to measure how well the neural network's predictions match the actual outputs.

### The training process includes:

Forward Propagation: Input data is passed through the network to generate predictions.

Loss Calculation: The difference between predicted and actual outputs is computed.(Binary Cross Entropy)

Backpropagation: The weights are updated based on the calculated loss to minimize prediction errors.