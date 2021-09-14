# This block of code includes classes for our neuarl network
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class layer_Dense:

    #layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        # creates weights as standard normal distribution
        # gaussian distribution, generates values to close 0. neg and pos
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # initialize those biases as "0" in 1D array
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

# input data has x and y cordinate
dense1 = layer_Dense(2, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(activation1.output)


# Let's see output of the first few samples:
print(activation1.output[:5])
print(activation2.output[:5])