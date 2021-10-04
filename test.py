


x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias


wx0 = x[0]*w[0]
wx1 = x[1]*w[1]
wx2 = x[2]*w[2]


print(wx0, wx1, wx2, b)

z = wx0 + wx1 + wx2 + b
print(z)


#relu activation function
y = max(z, 0)
print(f'relu: {y}')

#This is derivate for relu in backpropagation
drelu_dz = (1. if z > 0 else 0.)
print(drelu_dz)

dsum_dxw0 = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2
dsum_db = 1
drelu_db = drelu_dz * dsum_db


print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

