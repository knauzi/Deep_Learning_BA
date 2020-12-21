import matplotlib.pyplot as plt
import numpy as np

from src.neuronales_netz import NeuralNetwork
from src.aktivierungsfunktionen import Sigmoid, Relu, Softmax
from src.kostenfunktionen import QK, KE, BKE
import src.utils as utils

# Daten
X = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]])
Y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# Neuronales Netz
learning_rate = 0.01
epochs = 50000
batch_size = 10
nn = NeuralNetwork((2, 2, 3, 2), (Sigmoid, Sigmoid, Sigmoid), initialisation="random_normal")
history = nn.train(X, X, Y, Y, BKE, learning_rate, epochs, batch_size)

# plotte Kosten im Trainingsverlauf
plt.plot(history["Kosten-Training"])
plt.plot(history["Genauigkeit-Training"])
plt.xlabel("Durchlauf")
plt.ylabel("Kosten / Genauigkeit")
plt.show()

# plotte Entscheidungsgrenze
utils.plot_decision_boundary_2d(nn, X, Y)
