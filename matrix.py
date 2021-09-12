# import numpy as np

# a = [1, 2, 3]
# b = [2, 3, 4]

# a = np.array([a])
# b = np.array([b]).T


# print(f'Here is a array: {a} \nHere is b: {b}')
# print(f'Here is sum {np.dot(a,b)}')

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()