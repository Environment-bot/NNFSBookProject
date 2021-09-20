import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

p2_delta = 0.0001
x1 = 1
x2 = x1 + p2_delta # add delta
y1 = f(x1) # result at the derivation point
y2 = f(x2) # result at the other, close point
approximate_derivative = (y2-y1)/(x2-x1)
print(approximate_derivative)

