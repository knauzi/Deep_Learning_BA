import matplotlib.pyplot as plt

from src.neuronales_netz import ANN
from src.aktivierungsfunktionen import *
from src.kostenfunktionen import *
from src.utils import *

# Daten
X = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]])
Y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

# Neuronales Netz
learning_rate = 0.01
epochs = 50000
batch_size = 10
nn = ANN((2, 2, 3, 2), (Sigmoid, Sigmoid, Sigmoid), initialisation="random_normal")
history = nn.train(X, X, Y, Y, BKE, learning_rate, epochs, batch_size)

# plotte Kosten im Trainingsverlauf
plt.plot(history["Kosten-Training"])
plt.plot(history["Genauigkeit-Training"])
plt.xlabel("Durchlauf")
plt.ylabel("Kosten / Genauigkeit")
plt.show()

# plotte Entscheidungsgrenze
plot_decision_boundary_2d(nn, X, Y)