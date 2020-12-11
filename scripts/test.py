import numpy as np
from neuronales_netz import ANN
from aktivierungsfunktionen import *

# np.random.seed(6)
# X = np.random.randn(5,4)
# ann = ANN((5,4,3,1), (Relu, Relu, Sigmoid), None)
# A, caches = ann._forward_propagation(X)
# print(A)
# print(len(caches))

for i in reversed(range(2, 5)):
    print(i)