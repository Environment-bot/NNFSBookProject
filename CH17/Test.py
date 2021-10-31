import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import sine_data

nnfs.init()

X, y = sine_data()

plt.plot(X, y)
plt.show()