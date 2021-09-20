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

# common loss class
class Loss:
    def calculate(self, output, y):
        #calculate sample losses
        sample_losses = self.forward(output, y)

        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #return loss
        return data_loss

# cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of the samples in a batch
        samples = len(y_pred)

        # forces value to be between 1e-7 and 1 - 1e-7 
        # this gets noll value and over 1 value problems away
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values - 
        # only if categorial labels
        if len(y_true.shape) == 1:
            # if we have same 
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        #loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods




X, y = spiral_data(samples=100, classes=3)

# input data has x and y cordinate
dense1 = layer_Dense(2, 3)
dense2 = layer_Dense(3, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()


# # Make a forward pass of our training data through this layer
# dense1.forward(X)

# # Forward pass through activation func.
# # Takes in output from previous layer
# activation1.forward(dense1.output)

# # passing values to new dense layer 2
# dense2.forward(activation1.output)

# # Make a forward pass through activation function
# # it takes the output of second dense layer here
# activation2.forward(dense2.output)

# print(activation2.output[:5])

# loss = loss_function.calculate(activation2.output, y)
# # Let's see output of the first few samples:
# print('loss: ', loss)

# # accuracy

# predictions = np.argmax(activation2.output, axis=1)
# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
# accuracy = np.mean( predictions == y)

# print('acc: ', accuracy)

# indroduction optimization

lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # perform forward pass of the trainig data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
        'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()