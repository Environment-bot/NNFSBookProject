import numpy as np

# # Passed-in gradient from the next layer
# # for the purpose of this example we're going to use
# # an array of an incremental gradient values
# dvalues = np.array([[1., 1., 1.],
#                     [2., 2., 2.],
#                     [3., 3., 3.]])

# # We have 3 sets of weights - one set for each neuron
# # we have 4 inputs, thus 4 weights
# # recall that we keep weights transposed
# weights = np.array([[0.2, 0.8, -0.5, 1],
#                     [0.5, -0.91, 0.26, -0.5],
#                     [-0.26, -0.27, 0.17, 0.87]]).T
# print(weights)

# # Sum weights related to the given input multiplied by
# # the gradient related to the given neuron
# dinputs = np.dot(dvalues, weights.T)

# print(dinputs)

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# # an array of an incremental gradient values
# dvalues = np.array([[1., 1., 1.],
#                     [2., 2., 2.],
#                     [3., 3., 3.]])
# # We have 3 sets of inputs - samples
# inputs = np.array([[1, 2, 3, 2.5],
#                     [2., 5., -1., 2],
#                     [-1.5, 2.7, 3.3, -0.8]])
# # sum weights of given input
# # and multiply by the passed-in gradient for this neuron
# dweights = np.dot(inputs.T, dvalues)
# print(dweights)

# Example layer output
# z = np.array([[1, 2, -3, -4],
#                 [2, -7, -1, 3],
#                 [-1, 2, 5, -1]])
# dvalues = np.array([[1, 2, 3, 4],
#                     [5, 6, 7, 8],
#                     [9, 10, 11, 12]])
# # ReLU activation's derivative
# drelu = dvalues.copy()
# drelu[z <= 0] = 0



# print(drelu)

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
# dvalues = np.array([[1., 1., 1.],
#                     [2., 2., 2.],
#                     [3., 3., 3.]])
# # We have 3 sets of inputs - samples
# inputs = np.array([[1, 2, 3, 2.5],
#                     [2., 5., -1., 2],
#                     [-1.5, 2.7, 3.3, -0.8]])
# # We have 3 sets of weights - one set for each neuron
# # we have 4 inputs, thus 4 weights
# # recall that we keep weights transposed
# weights = np.array([[0.2, 0.8, -0.5, 1],
#                     [0.5, -0.91, 0.26, -0.5],
#                     [-0.26, -0.27, 0.17, 0.87]]).T
# print(weights)

# # One bias for each neuron
# # biases are the row vector with a shape (1, neurons)
# biases = np.array([[2, 3, 0.5]])

# # forward pass
# layer_outputs = np.dot(inputs, weights)+biases
# relu_outputs = np.maximum(0, layer_outputs)

# # Let's optimize and test backpropagation here
# # ReLU activation - simulates derivative with respect to input values
# # from next layer passed to current layer during backpropagation

# drelu = relu_outputs.copy()
# drelu[layer_outputs <= 0] = 0


# # Dense layer
# # dinputs - multiply by weights
# dinputs = np.dot(drelu, weights.T)
# #dweights - mutiply by inputs
# dweights = np.dot(inputs.T, drelu)

# # dbiases - sum values, do this over samples (first axis), keepdims
# # since this by default will produce a plain list -
# # we explained this in the chapter 4
# dbiases = np.sum(drelu, axis=0, keepdims=True)

# # Update parameters
# weights += -0.001 * dweights
# biases += -0.001 * dbiases

# print(weights)
# print(biases)

starting_learning_rate = 1.
learning_rate_decay = 0.02
for step in range(20):
    learning_rate = starting_learning_rate * \
    (1. / (1 + learning_rate_decay * step))
    print(learning_rate)