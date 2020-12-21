import os
import matplotlib.pyplot as plt

from src.neuronales_netz import NeuralNetwork, load_model
from src.aktivierungsfunktionen import Sigmoid, Relu, Softmax
from src.kostenfunktionen import QK, KE, BKE
import src.utils as utils

# Daten
n_samples = 500
split = 0.2
X, Y = utils.create_dataset_2d(n_samples=n_samples)
X_train, X_val, Y_train, Y_val = utils.get_train_val_split(X, Y, split=split)

# Neuronales Netz
learning_rate = 0.05
epochs = 300000
batch_size = 1
dropout_rate = None

nn = NeuralNetwork((2, 10, 10, 2), (Sigmoid, Sigmoid, Sigmoid), initialisation="random_normal")
# nn = NeuralNetwork((2, 10, 10, 2), (Relu, Relu, Softmax), initialisation="he")
# nn = NeuralNetwork((2, 50, 100, 50, 2), (Relu, Relu, Relu, Softmax), initialisation="he")
# nn = NeuralNetwork((2, 10, 10, 10, 10, 10, 10, 2), (Relu, Relu, Relu, Relu, Relu, Relu, Softmax), initialisation="he")

# history = nn.train(X_train, X_val, Y_train, Y_val, KE, learning_rate, epochs, batch_size, dropout_rate)
history = nn.train(X_train, X_val, Y_train, Y_val, QK, learning_rate, epochs, batch_size, dropout_rate)

output_path = os.path.dirname(os.path.abspath(__file__))
filename = "model_2d_data.json"
nn.save(output_path, filename)

# plotte Kosten im Trainingsverlauf
plt.plot(history["Kosten-Training"], label="Kosten-Training")
plt.plot(history["Kosten-Validierung"], label="Kosten-Validierung")
plt.xlabel("Durchlauf")
plt.ylabel("Kosten")
plt.legend()
plt.show()

# plotte Genauigkeit im Trainingsverlauf
plt.plot(history["Genauigkeit-Training"], label="Genauigkeit-Training")
plt.plot(history["Genauigkeit-Validierung"], label="Genauigkeit-Validierung")
plt.xlabel("Durchlauf")
plt.ylabel("Genauigkeit")
plt.legend()
plt.show()

# plotte Entscheidungsgrenze
utils.plot_decision_boundary_2d(nn, X, Y)

# test: load model
# model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_model.json")
# loaded_nn = load_model(model_file_path)
# utils.plot_decision_boundary_2d(loaded_nn, X, Y)
