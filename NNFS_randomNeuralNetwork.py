# This block of code includes classes for our neuarl network
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class layer_Dense:

    #layer initialization
    def __init__(self, n_inputs, n_neurons, 
                weight_regularizer_l1=0, weight_regularizer_l2=0,
                bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        # creates weights as standard normal distribution
        # gaussian distribution, generates values to close 0. neg and pos
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # initialize those biases as "0" in 1D array
        self.biases = np.zeros((1, n_neurons))

        # Set reqularization sternght
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1*dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2*self.weight_regularizer_l2 * \
                            self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1*dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2*self.bias_regularizer_l2 * \
                            self.biases

        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # create uniitialized array
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                    enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1,1)
            #calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                                np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                            single_dvalues)

class Activation_Sigmoid:
    # forward pass
    def forward(self, inputs):
        # safe input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs 
        self.output = 1/(1+np.exp(-inputs))
    # backward pass
    def backward(self, dvalues):
        # Derivative - calculation from output of the sigmoid function
        self.dinputs = dvalues * (1-self.output)*self.output

class Layer_Dropout:
    #init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1- rate

    def forward(self, inputs):
        #save input values
        self.inputs = inputs
        # Generate and save sclaed mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                             size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # gradient on values
        self.dinputs = dvalues * self.binary_mask

# common loss class
class Loss:
    def calculate(self, output, y):
        #calculate sample losses
        sample_losses = self.forward(output, y)

        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #return loss
        return data_loss

    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                                    np.sum(np.abs(layer.weights))
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                                    np.sum(layer.weights*layer.weights)
        # L1 regularization - bias
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                                    np.sum(np.abs(layer.biases))
        # L2 regularization - weights
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                                    np.sum(layer.biases*layer.biases)

        return regularization_loss

# cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of the samples in a batch
        samples = len(y_pred)
        # forces value to be between 1e-7 and 1 - 1e-7 
        # this gets noll value and over 1 value problems away
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
            )
        
        #loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward (self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        #number of labels in every sample
        # we'll use the first sample to count them
        labels = len(dvalues[0])

        # if labels are sparse, turn them into one-hot vector
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):
    #forward pass
    def forward(self, y_pred, y_true):
        # clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true*np.log(y_pred_clipped)+
                            (1-y_true)* np.log(1-y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

       


    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # If labes are one-hot encoded,
        # turn them into dscrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay =  decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1+self.decay * self.iterations))
    
    def update_params(self, layer):
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # current biases
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                            layer.dweights
            bias_updates = -self.current_learning_rate *\
                            layer.dbiases
        
        # Update weights and biases using either
        # Vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

         

    # Call once after any parameter updates  
    def post_update_params(self):
        self.iterations +=1

class AdaGrad:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay =  decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1+self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                            layer.dweights / \
                            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)  

    def post_update_params(self):
        self.iterations +=1

class Optmizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, 
                rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay =  decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1+self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                            (1 - self.rho) * layer.dbiases**2
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                        layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)  

    def post_update_params(self):
        self.iterations +=1

class Optmizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, 
                beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay =  decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1+self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # Update cache with squared current gradients
        layer.weight_momentums = self.beta_1 * \
                                layer.weight_momentums + \
                                (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                                layer.bias_momentums + \
                                (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
                                    (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                    (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                            (1 - self.beta_2) * layer.dweights**2  
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                                (1 - self.beta_2 ** (self.iterations + 1))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                        weight_momentums_corrected / \
                        (np.sqrt(weight_cache_corrected) +
                        self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                        self.epsilon)

    def post_update_params(self):
        self.iterations +=1
# Linear activation
class Activation_Linear:
    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

# Mean squared error loss
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        # calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1,1)

# input data has x and y cordinate
dense1 = layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
dense2 = layer_Dense(64, 1)
activation1 = Activation_ReLU()
# dropout1 = Layer_Dropout(0.1)
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossentropy()
# loss_function = Activation_Softmax_Loss_CategoricalCrossentropy()
# optimizer = Optimizer_SGD(learning_rate=1, decay=1e-3, momentum=0.9)
# optimizer = AdaGrad(learning_rate=1, decay=1e-4)
# optimizer = Optmizer_RMSprop(learning_rate=0.02,decay=1e-4, rho=0.999)
optimizer = Optmizer_Adam(decay=5e-7)



for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    # dropout1.forward(activation1.output)

    
    dense2.forward(activation1.output)
    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.output)
    # loss = loss_function.forward(dense2.output, y)
    data_loss = loss_function.calculate(activation2.output, y)

    # Calculate regularization penalty
    regularization_loss = \
            loss_function.regularization_loss(dense1) + \
            loss_function.regularization_loss(dense2)

    # Calculate overall loss
    loss = data_loss + regularization_loss
    # Let's print loss value
    # print('loss:', loss)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    # predictions = np.argmax(loss_function.output, axis=1)
    # if len(y.shape) == 2:
    #     y = np.argmax(y, axis=1)
    # accuracy = np.mean(predictions==y)



    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, '+
                f'loss: {loss:.3f} (' +
                f'data_loss: {data_loss:.3f}, ' +
                f'reg_loss: {regularization_loss:.3f}), ' +
                f'lr: {optimizer.current_learning_rate}')
   

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    # dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
# predictions = np.argmax(loss_function.output, axis=1)
# if len(y_test.shape) == 2:
#     y_test = np.argmax(y_test, axis=1)
# accuracy = np.mean(predictions == y_test)

# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

