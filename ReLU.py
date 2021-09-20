import numpy as np

# softmax_outputs = np.array([[0.7, 0.1, 0.2],
# [0.1, 0.5, 0.4],
# [0.02, 0.9, 0.08]])
# class_targets = np.array([1, 0, 0],
#                         [0, 1, 0],
#                         [1, 0, 0])
# print(len(class_targets.shape))


# if len(class_targets.shape) == 1:
#     # first dimensions can be passed as range[1,2,3]
#     # second value choses which value is taken from that array
#     # this works only with output as 2D and class target as 1D array
#     correct_confidences = softmax_outputs[
#     range(len(softmax_outputs)),
#     class_targets
#     ]

# elif len(class_targets.shape) == 2:
#     #this code gets output as 2D array as well as class targets
#     # then it goes sum function between all of the values
#     # [0.7, 0.1, 0.2]*[1, 0, 0] is first multiplyed by targets values 
#     # and then it is summed -> 0,7*1+0*0,1... => 0,7
#     correct_confidences = np.sum(
#     softmax_outputs*class_targets,
#     axis=1)

# correct_confidences = np.array([0.7, 0.5, 0.9])

# print(-np.log(0.9999999999999))

# Probabilities of 3 samples
# softmax_outputs = np.array([[0.7, 0.2, 0.1],
# [0.5, 0.1, 0.4],
# [0.02, 0.9, 0.08]])
# # Target (ground-truth) labels for 3 samples
# class_targets = np.array([0, 1, 1])

# predictions  = np.argmax(softmax_outputs, axis=1)
# print(predictions)

# if len(class_targets.shape) == 2:
#     class_targets = np.argmax(softmax_outputs, axis=1)
# print(predictions == class_targets)
# accuracy = np.mean(predictions == class_targets)
# print('acc :', accuracy)

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()

X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()



