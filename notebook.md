<h1>This notebook is used to clarify NNFS books topics</h1>

# Inputs

Inputs are on of the key topics in NNFS book. First we have used to pass to neurons only one set of inputs. To get better output from the neural network it needs to be used batch of inputs which is called **samples**. 

These **samples** contains multiple input sets for that layer of neurons. These samples are then calculated at the same weights. This helps to adjust correctly the network

# Weight and bias initialization

Random initialization can be used when creating neurons to the network and this proach has been used on this course too. 

But for reminding there can be used pre trained values for network as it does not start from scratch or there could be generated some rules for creating those numbers. More about this can be found from **dense layer class** section in nnfs book (page 66->).

# ReLU aka Rectified Linear Unit

with relu function we can adjust when single neuron fires it's output. For example we have neagtive value as input we could determinate that it cannot pass anything true until it is positive value. With ReLU we can do this most easyest way.

## One neuron

By increasing bias (pos), we’re making this neuron activate earlier.

With a negative weight and this single neuron, the function has become a question of when this
neuron deactivates.


## with two neurons

The second neurons bias now horizontally adjusts the output and weight can adjust slopes direction agen.

More on this topic on pages 88-97.

# Softmax activation function

This is used for clasification layer aka "output" layer to choose correct output. this clasificators output is called **confidence score** 

This is prosent number which describes each predictions **confidence score**

# Cross-entropy loss

with cross entropy we can calculte from multiple outputs what is our overall loss as well our summed loss in a batch

```python
softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])
class_targets = np.array([1, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0])
print(len(class_targets.shape))


if len(class_targets.shape) == 1:
    # first dimensions can be passed as range[1,2,3]
    # second value choses which value is taken from that array
    # this works only with output as 2D and class target as 1D array
    correct_confidences = softmax_outputs[
    range(len(softmax_outputs)),
    class_targets
    ]

elif len(class_targets.shape) == 2:
    #this code gets output as 2D array as well as class targets
    # then it goes sum function between all of the values
    # [0.7, 0.1, 0.2]*[1, 0, 0] is first multiplyed by targets values 
    # and then it is summed -> 0,7*1+0*0,1... => 0,7
    correct_confidences = np.sum(
    softmax_outputs*class_targets,
    axis=1)
```

# Accuracy calcultaion (129-130)

we can calculate accuracy as simple as taking output from our model and compering it to our ground truth "y"

First we will take our softmax outputs and convert it to ones and zeros regarding which value has highest confidence from softwmax. Then it is compared our "y" values and those values comparations mean is then our accuracy.

