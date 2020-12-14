import matplotlib.pyplot as plt

from src.neuronales_netz import ANN
from src.aktivierungsfunktionen import *
from src.kostenfunktionen import *
from src.utils import *

# Daten
n_samples = 1000
split = 0.2
X, Y = create_dataset_2d(n_samples=n_samples)
X_train, X_val, Y_train, Y_val = get_train_val_split(X, Y, split=split)

# Neuronales Netz
learning_rate = 0.01
epochs = 10000
batch_size = 32
nn = ANN((2, 10, 10, 2), (Relu, Relu, Softmax), initialisation="he")
history = nn.train(X_train, X_val, Y_train, Y_val, KE, learning_rate, epochs, batch_size)

# plotte Kosten im Trainingsverlauf
plt.plot(history["Kosten-Training"])
plt.plot(history["Kosten-Validierung"])
plt.xlabel("Durchlauf")
plt.ylabel("Kosten")
plt.show()

# plotte Genauigkeit im Trainingsverlauf
plt.plot(history["Genauigkeit-Training"])
plt.plot(history["Genauigkeit-Validierung"])
plt.xlabel("Durchlauf")
plt.ylabel("Genauigkeit")
plt.show()

# plotte Entscheidungsgrenze
plot_decision_boundary_2d(nn, X, Y)