import os
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import trange
import json

import src.initialisierung as init
from src.aktivierungsfunktionen import Sigmoid, Relu, Softmax
from src.kostenfunktionen import QK, KE, BKE
import src.utils as utils


class NeuralNetwork:
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

        assert len(layer_dims) - 1 == len(activations), \
            "Anzahl der Schichten und Aktivierungsfunktionen stimmen nicht " \
            "überein: {} -1 != {}".format(len(layer_dims), len(activations))

        self.n_layers = len(layer_dims)
        self.layer_dims = layer_dims

        # initialisiere Parameter nach der vorgegebenen Methode
        if initialisation == "random_normal":
            self.parameters = init.init_random_normal(layer_dims)
        elif initialisation == "xavier":
            self.parameters = init.init_xavier_uniform(layer_dims)
        elif initialisation == "he":
            self.parameters = init.init_he_uniform(layer_dims)
        elif initialisation == "zero":
            self.parameters = init.init_zeros(layer_dims)
        elif initialisation == "constant":
            self.parameters = init.init_constant(layer_dims)
        elif initialisation == "load":
            pass
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

    def _forward_propagation_with_dropout(self, X, dropout_rate):
        """
            Implementierung der Feedforward-Logik mit Dropout
            Speichert alle Zwischenergebnisse der einzelnen Schichten

            Args:
                X: Input-Daten, numpy array der Form (Input-Größe, Anzahl Datenpunkte)
                dropout_rate: Anteil der Neuronen in jeder versteckten Schicht, die fallen gelassen werden

            Returns:
                Z: lineare Zwischenergebnisse aller Schichten
                A: Aktivierungen aller Schichten (Nummerierung entspricht Beispiel von oben)
                D: Dropout-Masken für versteckte Schichten (Nummerierung entspricht Beispiel von oben)
        """

        Z = {}  # Input hat keinen linaren Anteil
        A = {1: X}  # Aktivierung der ersten Schicht ist der Input
        D = {}

        for l in range(1, self.n_layers - 1):

            # linearer Anteil der aktuellen Schicht (Formel: Z = W * A + b)
            Z[l+1] = np.dot(self.parameters["W"+str(l+1)], A[l]) + self.parameters["b"+str(l+1)]

            # nicht-linearer Anteil der aktuelle Schicht (Formel: A = sigma(Z))
            A[l+1] = self.activations[l+1].forward(Z[l+1])

            # TODO Dropout richtig implementiert ?
            # dropout
            # binäre Maske und Skalierung in einer Matrix gespeichert
            D[l+1] = (np.random.rand(*A[l+1].shape) < dropout_rate) / dropout_rate
            A[l+1] = np.multiply(A[l+1], D[l+1])

        # kein dropout in Ausgabe-Schicht
        L = self.n_layers
        Z[L] = np.dot(self.parameters["W"+str(L)], A[L-1]) + self.parameters["b"+str(L)]
        A[L] = self.activations[L].forward(Z[L])

        return Z, A, D

    def _get_deltaL(self, AL, ZL, Y):
        """
            Berechnet den Error in der Ausgabe-Schicht in Abhängigkeit von der Kostenfunktion und
            Aktivierungsfunktion in der Ausgabe-Schicht

            Args:
                AL: Ausgaben des neuronalen Netzes, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                ZL: linearer Anteil der Ausgabe-Schicht, Dimension (Anz. Ausgaben, Anz. Datenpunkte)
                Y: erwartete Ausgaben, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)

            Returns:
                deltaL: Fehler in der Ausgabe-Schicht, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
        """

        # Kostenfunktionen alle unskaliert !!! (keine Mittelwertbildung)
        # TODO Skalierung !?!? -> erst bei Parameter-Update??

        # 1. Fall: quadratische Kostenfunktion und Sigmoid-Aktivierungsfunktion
        if self.cost_function == QK and self.activations[self.n_layers] == Sigmoid:
            deltaL = (AL - Y) * self.activations[self.n_layers].backward(ZL)
        # 2. Fall: binäre Kreuzentropie-Kostenfunktion und Sigmoid-Aktivierungsfunktion
        elif self.cost_function == BKE and self.activations[self.n_layers] == Sigmoid:
            deltaL = (AL - Y)
        # 3. Fall: Kreuzentropie-Kostenfunktion und Softmax-Aktivierungsfunktion
        elif self.cost_function == KE and self.activations[self.n_layers] == Softmax:
            deltaL = (AL - Y)
        # 4. Fall: Kreuzentropie-Kostenfunktion und Sigmoid-Aktivierungsfunktion (nicht gezeigt in Arbeit)
        elif self.cost_function == KE and self.activations[self.n_layers] == Softmax:
            deltaL = (AL - Y)
        else:
            raise Exception("Unbekannte Kombination von Kostenfunktion und Aktivierungsfunktion in der"
                            " Ausgabe-Schicht!")

        return deltaL

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
        ZL = Z[self.n_layers]
        delta = {self.n_layers: self._get_deltaL(AL, ZL, Y)}

        # Fehler aller restlichen Schichten
        for l in reversed(range(2, self.n_layers)):
            delta[l] = self.activations[l].backward(Z[l]) * \
                       np.dot(np.transpose(self.parameters["W"+str(l+1)]), delta[l+1])

        return delta

    def _backward_propagation_with_dropout(self, Z, A, Y, D):
        """
            Implementierung der Backpropagation

            Args:
                Z: linearer Anteil aller Zwischenschichten
                A: Aktivierungen aller Zwischenschichten
                Y: Erwarteter Output
                D: Dropout-Masken für versteckte Schichten (Nummerierung entspricht Beispiel von oben)

            Return:
                delta: Fehler in jeder Schicht (Nummerierung wie im Beispiel oben)
        """

        # Fehler in der Output-Schicht (vgl. Kapitel Backpropagation)
        AL = A[self.n_layers]
        ZL = Z[self.n_layers]
        delta = {self.n_layers: self._get_deltaL(AL, ZL, Y)}

        # TODO Dropout richtig implementiert ?
        # Fehler aller restlichen Schichten
        # mit dropout !
        for l in reversed(range(2, self.n_layers)):
            delta[l] = self.activations[l].backward(Z[l]) * \
                       np.dot(np.transpose(self.parameters["W" + str(l + 1)]), delta[l + 1])
            delta[l] = np.multiply(delta[l], D[l])

        return delta

    def _update_parameters(self, delta, A):
        """
            Aktualisierung aller Parameter mittels der berechneten Fehler in jeder Schicht (delta)

            Args:
                delta: Fehler in jeder Schicht (Nummerierung wie in Beispiel oben)
                A: Aktivierungen aller Zwischenschichten
        """

        # TODO batch scaling -> done
        n = A[self.n_layers].shape[1]  # Anzahl Datenpunkte (in Durchlauf, insb. Bacht-Größe)
        for l in reversed(range(2, self.n_layers + 1)):

            self.parameters["W"+str(l)] = self.parameters["W"+str(l)] - \
                                          self.learning_rate * np.dot(delta[l], np.transpose(A[l-1])) \
                                          / n

            self.parameters["b"+str(l)] = self.parameters["b"+str(l)] - \
                                          self.learning_rate * np.sum(delta[l], axis=-1, keepdims=True) \
                                          / n

    def train(self, X_train, X_val, Y_train, Y_val, cost_function, learning_rate, epochs, batch_size,
              dropout_rate=None):
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
                dropout_rate: Anteil der Neuronen in jeder versteckten Schicht, die fallen gelassen werden
                              -> None, falls kein Dropout stattfinden soll

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

            # forward -> backward -> update
            if dropout_rate is None:
                Z, A = self._forward_propagation(x_train)
                delta = self._backward_propagation(Z, A, y_train)
            else:
                Z, A, D = self._forward_propagation_with_dropout(x_train, dropout_rate)
                delta = self._backward_propagation_with_dropout(Z, A, y_train, D)
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
                predictions_train = utils.get_one_hot(np.argmax(A_train[self.n_layers], axis=0), n_classes)
                predictions_val = utils.get_one_hot(np.argmax(A_val[self.n_layers], axis=0), n_classes)
                accuracy_train = utils.get_accuracy(Y_train, predictions_train)
                accuracy_val = utils.get_accuracy(Y_val, predictions_val)
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

    def save(self, output_path, filename):
        """
            Speichert das neuronale Netz mit allen Parametern in einer .json Datei

            Args:
                output_path: Dateipfad, wo das neuronale Netz gespeichert werden soll
                filename: Name der .json Datei (z.B. test.model.json)
        """

        model_dict = {}
        model_dict["layer_dims"] = self.layer_dims
        model_dict["n_layers"] = self.n_layers
        model_dict["activations"] = [activation.__name__ for _, activation in self.activations.items()]
        model_dict["cost_function"] = self.cost_function.__name__
        model_dict["learning_rate"] = self.learning_rate

        model_dict["parameters"] = {}
        for l in range(1, self.n_layers):
            model_dict["parameters"]["W" + str(l + 1)] = self.parameters["W" + str(l + 1)].tolist()
            model_dict["parameters"]["b" + str(l + 1)] = self.parameters["b" + str(l + 1)].tolist()

        path = os.path.join(output_path, filename)
        with open(path, "w") as fp:
            json.dump(model_dict, fp)


def load_model(model_file_path):
    """
        Rekonstruiert ein neuronales Netz mit allen Parametern von einer .json Datei

        Args:
            model_file_path: kompletter Dateipfad der .json Datei (z.B. mein/ordner/test.model.json)

        Returns:
            model: Instanz der Klasse Neural_Network
    """

    with open(model_file_path, "r") as fp:
        model_dict = json.load(fp)

    n_layers = model_dict["n_layers"]
    layer_dims = model_dict["layer_dims"]
    learning_rate = model_dict["learning_rate"]

    activation_dict = {"Sigmoid": Sigmoid,
                       "Relu": Relu,
                       "Softmax": Softmax}
    activations = [activation_dict[activation] for activation in model_dict["activations"]]

    cost_dict = {"QK": QK,
                 "KE": KE,
                 "BKE": BKE}
    cost_function = cost_dict[model_dict["cost_function"]]

    parameters = {}
    for l in range(1, n_layers):
        parameters["W" + str(l + 1)] = np.array(model_dict["parameters"]["W" + str(l + 1)])
        parameters["b" + str(l + 1)] = np.array(model_dict["parameters"]["b" + str(l + 1)])

    model = NeuralNetwork(layer_dims, activations, initialisation="load")
    model.cost_function = cost_function
    model.parameters = parameters
    model.learning_rate = learning_rate

    return model
