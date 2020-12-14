import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from src.initialisierung import *
from src.aktivierungsfunktionen import *
from src.kostenfunktionen import *
from src.utils import *


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
        elif initialisation == "constant":
            self.parameters = init_constant(layer_dims)
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

        # Fehler in der Output-Schicht (vgl. Kapitel Backpropagation)
        AL = A[self.n_layers]
        deltaL = None
        # 1. Fall: quadratische Kostenfunktion und Sigmoid-Aktivierungsfunktion
        if self.cost_function == QK and self.activations[self.n_layers] == Sigmoid:
            deltaL = (AL - Y) * self.activations[self.n_layers].backward(Z[self.n_layers])
        # 2. Fall: binäre Kreuzentropie-Kostenfunktion und Sigmoid-Aktivierungsfunktion
        elif self.cost_function == BKE and self.activations[self.n_layers] == Sigmoid:
            deltaL = (AL - Y)
        # 3. Fall: Kreuzentropie-Kostenfunktion und Softmax-Aktivierungsfunktion
        elif self.cost_function == KE and self.activations[self.n_layers] == Softmax:
            deltaL = (AL - Y)
        else:
            raise Exception("Unbekannte Kombination von Kostenfunktion und Aktivierungsfunktion in der"
                            " Ausgabe-Schicht!")

        delta = {self.n_layers: deltaL}

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

        for l in reversed(range(2, self.n_layers + 1)):
            self.parameters["W"+str(l)] = self.parameters["W"+str(l)] - \
                                          self.learning_rate * np.dot(delta[l], A[l-1].T)
            self.parameters["b"+str(l)] = self.parameters["b"+str(l)] - \
                                          self.learning_rate * np.sum(delta[l], axis=-1, keepdims=True)

    def train(self, X_train, X_val, Y_train, Y_val, cost_function, learning_rate, epochs, batch_size):
        """
            Trainierung des neuronalen Netzes auf den gegebenen Daten mittles des stochastischen
            Gradientverfahrens

            Args:
                X_train: Trainingseingaben, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                X_val: Validierungseingaben, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                Y_train: Erwartete Trainingsausgaben ({0,1}),
                         Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                Y_val: Erwartete Validierungsausgaben ({0,1}),
                       Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                cost_function: Kostenfunktion
                learning_rate: Lernrate
                epochs: Anzahl an Iterationen
                batch_size: Mini-Batch Größe

            Returns:
                history: Dictionary mit Kosten und Genauigkeit
        """

        self.learning_rate = learning_rate
        self.cost_function = cost_function

        n_classes = X_train.shape[0]
        n_train = X_train.shape[1]

        costs_train = []
        costs_val = []
        accuracies_train = []
        accuracies_val = []

        for i in (t := trange(epochs)):

            # Ziehen mit Zurücklegen
            # TODO Ziehen ohne Zurücklegen implementieren
            sample = np.random.randint(n_train, size=batch_size)
            x_train = X_train[:, sample]
            y_train = Y_train[:, sample]

            Z, A = self._forward_propagation(x_train)
            delta = self._backward_propagation(Z, A, y_train)
            self._update_parameters(delta, A)

            # Berechne die Kosten und Genauigkeit über alle Daten mit den aktualisierten Parametern
            # und gib diese auf der Konsole aus; alle 10 Epochen
            if (i % 10) == 0:
                _, A_train = self._forward_propagation(X_train)
                _, A_val = self._forward_propagation(X_val)

                # Kosten
                cost_train = self.cost_function.compute(A_train[self.n_layers], Y_train)
                cost_val = self.cost_function.compute(A_val[self.n_layers], Y_val)
                costs_train.append(cost_train)
                costs_val.append(cost_val)

                # Genauigkeit
                predictions_train = get_one_hot(np.argmax(A_train[self.n_layers], axis=0), n_classes)
                predictions_val = get_one_hot(np.argmax(A_val[self.n_layers], axis=0), n_classes)
                accuracy_train = get_accuracy(Y_train, predictions_train)
                accuracy_val = get_accuracy(Y_val, predictions_val)
                accuracies_train.append(accuracy_train)
                accuracies_val.append(accuracy_val)

                # gib Kosten und Genauigkeit auf der Konsole aus
                t.set_description("Kosten-Train: {:0.2f}; "
                                  "Genauigkeit-Train: {:0.2f}; "
                                  "Kosten-Val: {:0.2f}; "
                                  "Genauigkeit-Val: {:0.2f}".format(cost_train, accuracy_train,
                                                                    cost_val, accuracy_val))

        history = {"Kosten-Training": costs_train,
                   "Kosten-Validierung": costs_val,
                   "Genauigkeit-Training": accuracies_train,
                   "Genauigkeit-Validierung": accuracies_val}

        return history

    def predict(self, X):
        """
            Inferenz des neuronalen Netzes auf den Eingaben

            Args:
                X: Eingaben, Dimension (Anzahl Ausaben, Anzahl Datenpunkte)

            Returns:
                AL: Ausgaben des neuronalen Netzes, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
        """

        A, _ = self._forward_propagation(X)
        AL = A[self.n_layers]

        return AL

    def save(self, output_path):
        pass
