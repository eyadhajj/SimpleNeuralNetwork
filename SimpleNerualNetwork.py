# Imports required
import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
synaptic_weights = (2 * np.random.random((3, 1))) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

for iteration in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    # Adjust weights using gradient descent
    adjustments = np.dot(input_layer.T, error * sigmoid_derivative(outputs))
    
    synaptic_weights += adjustments

print(f'Synaptic weights after training: {synaptic_weights}')
print(f'Outputs after training: {outputs}')
