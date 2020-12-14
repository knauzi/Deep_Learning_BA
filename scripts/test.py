import numpy as np
from src.neuronales_netz import ANN
from src.aktivierungsfunktionen import *
import src.utils as utils
import matplotlib.pyplot as plt

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

# a = np.array([[1, 4], [1, 3]])
# print(a)
# print(np.sum(a, axis=-1, keepdims=True))

# # Daten
# from src.utils import plot_data_2d
# import matplotlib.pyplot as plt
X = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]])
Y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
# plot_data_2d(X, Y)

# X, Y = utils.create_dataset_2d()
# utils.plot_data_2d(X, Y)


# # Set min and max values and give it some padding
# x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
# y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
# h = 0.01
#
# # Generate a grid of points with distance h between them
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Predict the function value for the whole grid
# ann = ANN((2, 2, 3, 2), (Sigmoid, Sigmoid, Sigmoid), initialisation="random_normal")
# Z = ann.predict(np.transpose(np.c_[xx.ravel(), yy.ravel()]))
# Z = np.argmax(Z, axis=0)
# print(Z)
# Z = Z.reshape(xx.shape)
#
# # Plot the contour and training examples
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
# plt.ylabel('x2')
# plt.xlabel('x1')
# y = np.argmax(Y, axis=0)
# plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
# plt.show()

from src.utils import *
X_train, X_val, Y_train, Y_val = get_train_val_split(X, Y, split=0.2)
print(X_train)
print(X_val)
print(Y_train)
print(Y_val)
