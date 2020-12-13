import numpy as np
from src.neuronales_netz import ANN
from src.aktivierungsfunktionen import *

# np.random.seed(6)
# X = np.random.randn(5,2)
# ann = ANN((5,4,3,4), (Relu, Relu, Softmax))
# Z, A = ann._forward_propagation(X)
# A = A[ann.n_layers]
# print(A[[0],[1]])
# print(A)
# print(Softmax.backward(A))

# for i in reversed(range(2, 5)):
#     print(i)

# x = np.random.random(size=(5,1))
# print(Softmax.forward(x))
# print(np.sum(Softmax.forward(x)))

# a = np.eye(2)[np.array([0, 1, 1, 1, 0])]
# print(a)
# print(np.transpose(a))

a = np.array([[1, 4], [1, 3]])
print(a)
print(np.sum(a, axis=-1, keepdims=True))