import numpy as np
from src.neuronales_netz import ANN
from src.aktivierungsfunktionen import *

np.random.seed(6)
X = np.random.randn(5,2)
ann = ANN((5,4,3,4), (Relu, Relu, Softmax))
Z, A = ann._forward_propagation(X)
A = A[ann.n_layers]
print(Softmax.backward(A))

# for i in reversed(range(2, 5)):
#     print(i)

# x = np.random.random(size=(5,1))
# print(Softmax.forward(x))
# print(np.sum(Softmax.forward(x)))