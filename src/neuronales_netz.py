import numpy as np
import matplotlib.pyplot as plt

from .initialisierung import init_random_normal, init_xavier_uniform, init_he_uniform, init_zeros
from .aktivierungsfunktionen import Sigmoid, Relu, Softmax
from .kostenfunktionen import MSE, BCE


class ANN:
    """ Klasse, die ein künstliches neuronales Netz implementiert. """

    def __init__(self, layer_dims, activations, initialisation="random_normal"):
        """
            Initialisierung des neuronalen Netzes

            Args:
                layer_dims: Liste mit Dimensionen aller Schichten der Länge n_layers 
                activations: Liste mit nicht-linearen Aktivierungsfunktionen aller Schichten
                initialisation: String mit anzuwendender Initialisierungsmethode

            Beispiel:
                - 4 Inputs
                - 5 Neuronen in Zwischenschicht 1
                - 8 Neuronen in Zwischenschicht 2
                - 1 Output

                Schichten(Nummerierung)   ->  (1, 2,    3,    4)
                layer_dims                =   (4, 5,    8,    1)
                activations               =   (   Relu, Relu, Sigmoid)
        """

        self.n_layers = len(layer_dims)
        self.layer_dims = layer_dims

        # initialisiere Parameter nach der vorgegebenen Methode
        if initialisation == "random_normal":
            self.parameters = init_random_normal(layer_dims)
        elif initialisation == "xavier":
            self.parameters = init_xavier_uniform(layer_dims)
        elif initialisation == "he":
            self.parameters = init_he_uniform(layer_dims)
        elif initialisation == "zero":
            self.parameters = init_zeros(layer_dims)
        else:
            raise AttributeError("Unbekannte Initialisierungsmethode übergeben!")

        # speichere Aktivierungsfunktionen passend zur Nummerierung der Schichten
        self.activations = {}
        for l in range(self.n_layers - 1):
            self.activations[l+2] = activations[l]

        # als Parameter der Funktion "train" übergeben
        self.cost_function = None
        self.learning_rate = None

    def _forward_propagation(self, X):
        """
            Implementierung der Feedforward-Logik
            Speichert alle Zwischenergebnisse der einzelnen Schichten

            Args:
                X: Input-Daten, numpy array der Form (Input-Größe, Anzahl Datenpunkte)

            Returns:
                Z: lineare Zwischenergebnisse aller Schichten
                A: Aktivierungen aller Schichten (Nummerierung entspricht Beispiel von oben)
        """

        Z = {}  # Input hat keinen linaren Anteil
        A = {1: X}  # Aktivierung der ersten Schicht ist der Input

        for l in range(1, self.n_layers):

            # linearer Anteil der aktuellen Schicht (Formel: Z = W * A + b)
            Z[l+1] = np.dot(self.parameters["W"+str(l+1)], A[l]) + self.parameters["b"+str(l+1)]

            # nicht-linearer Anteil der aktuelle Schicht (Formel: A = sigma(Z))
            A[l+1] = self.activations[l+1].forward(Z[l+1])

        return Z, A

    def _backward_propagation(self, Z, A, Y):
        """
            Implementierung der Backpropagation
            
            Args:
                Z: linearer Anteil aller Zwischenschichten
                A: Aktivierungen aller Zwischenschichten
                Y: Erwarteter Output

            Return:
                delta: Fehler in jeder Schicht (Nummerierung wie im Beispiel oben)
        """

        # Fehler in der Output-Schicht
        delta = {self.n_layers: self.cost_function.prime(A[self.n_layers], Y) *
                                self.activations[self.n_layers].backward(Z[self.n_layers])}

        # Fehler aller restlichen Schichten
        for l in reversed(range(2, self.n_layers)):
            delta[l] = self.activations[l].backward(Z[l]) * \
                       np.dot(self.parameters["W"+str(l+1)].T, delta[l+1])

        return delta

    def _update_parameters(self, delta, A):
        """
            Aktualisierung aller Parameter mittels der berechneten Fehler in jeder Schicht (delta)

            Args:
                delta: Fehler in jeder Schicht (Nummerierung wie in Beispiel oben)
                A: Aktivierungen aller Zwischenschichten
        """

        n_examples = A[1].shape[1]  # Anzahl an Trainings-Beispielen
        for l in reversed(range(2, self.n_layers + 1)):
            self.parameters["W"+str(l)] = self.parameters["W"+str(l)] - \
                                          self.learning_rate * (1 / n_examples) * np.dot(delta[l], A[l-1].T)
            self.parameters["b"+str(l)] = self.parameters["b"+str(l)] - \
                                          self.learning_rate * np.mean(delta[l], axis=0, keepdims=True)

    def train_stochastic(self, X, Y, cost_function, learning_rate, n_iter, print_cost=False):
        """
            Trainierung des neuronale Netzes auf den gegebenen Daten mittles des stochastischen
            Gradientverfahrens

            Args:
                X: Input
                Y: Erwarteter Output
                cost_function: Kostenfunktion
                learning_rate: Lernrate
                n_iter: Anzahl an Iterationen
                print_cost: (boolean) Sollen die Kosten in der Konsole ausgegeben werden?: JA/NEIN

            Returns:
                costs: Liste mit Kosten im Verlauf des Trainings (leer, wenn plot_cost False ist)
        """

        self.learning_rate = learning_rate
        self.cost_function = cost_function
        costs = []
        n_examples = X.shape[1]

        for i in range(n_iter):
            k = np.random.randint(n_examples)
            x = X[:, [k]]
            y = Y[:, [k]]
            Z, A = self._forward_propagation(x)
            delta = self._backward_propagation(Z, A, y)
            self._update_parameters(delta, A)

            # Berechne die Kosten über alle Daten mit den aktualisierten Parametern und gib diese
            # auf der Konsole aus, wenn print_cost=true
            # TODO ändere zu history und speicher noch andere Daten zum Training
            if (i % 100) == 0:
                _, A = self._forward_propagation(X)
                cost = self.cost_function.compute(A[self.n_layers], Y)
                costs.append(cost)
                if print_cost:
                    print("Kosten nach Iteration {}: {}".format(i, cost))

        return costs

    def save(self, output_path):
        pass


if __name__ == "__main__":

    # Daten
    X_train = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                        [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]])
    Y_train = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    # Neuronales Netz
    learning_rate = 0.05
    n_iter = 1000000
    nn = ANN((2, 2, 3, 2), (Sigmoid, Sigmoid, Sigmoid))
    costs = nn.train_stochastic(X_train, Y_train, MSE, learning_rate, n_iter, print_cost=False)

    # plotte Kosten im Trainingsverlauf
    iter_numbers = np.arange(0, n_iter, 100)
    plt.plot(iter_numbers, costs)
    plt.xlabel("Iter Number")
    plt.ylabel("Cost")
    plt.show()
