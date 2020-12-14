import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons


def get_one_hot(targets, n_classes):
    """
        Verwandelt Vektor in one-hot Darstellung

        Args:
            targets: 1-D Array mit Klassen
            n_classes: Gesamtzahl der Klassen

        Returns:
            one_hot: one-hot Darstellung
    """

    one_hot = np.eye(n_classes)[targets]
    return np.transpose(one_hot)


def get_accuracy(Y, pred):
    """
        Berechnet die Genauigkeit des neuronalen Netzes

        Args:
            Y: erwartete Ausgabe, Dimension (Anzahl Ausgaben, Anzahl Anzahl Datenpunkte)
            pred: one-hot Vorhersagen des neuronalen Netzes,
                  Dimension (Anzahl Ausgaben, Anzahl Anzahl Datenpunkte)

        Returns:
            accuarcy: Genauigkeit
    """

    n_examples = Y.shape[1]
    n_true = np.sum(np.array([np.array_equal(Y[:, j], pred[:, j])
                              for j in range(n_examples)]))
    accuracy = np.squeeze(n_true) / n_examples

    return accuracy


def plot_data_2d(X, Y, limit=None, cmap="Spectral"):
    """
        Erstellt Grafik mit Datenpunkten in 2D

        Args:
            X: Eingaben, Dimension (2, Anzahl Datenpunkte)
            Y: Ausgaben, Dimension (2, Anzahl Datenpunkte)
            limit: Tuple (min, max) Beschränkung der Achsen
            cmap: Matplotlib Farbmap
    """

    # sortiere Daten in die beiden Klassen
    # n_examples = X.shape[1]
    # indices1 = [k for k in range(n_examples) if np.array_equal(Y[:, k], np.array([1, 0]))]
    # indices2 = [k for k in range(n_examples) if np.array_equal(Y[:, k], np.array([0, 1]))]
    # X1, X2 = X[:, indices1], X[:, indices2]

    # erstelle Grafik
    plt.ylabel('x2')
    plt.xlabel('x1')
    if limit is not None:
        low, high = limit
        plt.xlim(low, high)
        plt.ylim(low, high)
    # plt.scatter(X1[0, :], X1[1, :], marker="o", cmap=cmap)
    # plt.scatter(X2[0, :], X2[1, :], marker="s", cmap=cmap)
    y = np.argmax(Y, axis=0)
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.get_cmap(cmap))
    plt.show()


def get_train_val_split(X, Y, split=0.2):
    """
        Teilt die Eingaben und Ausgaben zufällig in Trainings- und Validierungsdaten auf

    Args:
        X: Eingaben, Dimension(Anzahl Ausgaben, Anzahl Datenpunkte)
        Y: Eingaben, Dimension(Anzahl Ausgaben, Anzahl Datenpunkte)
        split: Prozent, wie viele Datenpunkte in Validierungsdatensatz kommen
    Returns:
        X_train: Trainingseingaben
        X_val: Validierungseingaben
        Y_train: Trainingsausgaben
        Y_val: Validierungsausgaben
    """

    n_samples = X.shape[1]
    n_train = int((1 - split) * n_samples)

    # Erzeuge Vektor mit Inidizes und mische diese
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Teile Daten
    X_train, X_val = X[:, indices[:n_train]], X[:, indices[n_train:]]
    Y_train, Y_val = Y[:, indices[:n_train]], Y[:, indices[n_train:]]

    return X_train, X_val, Y_train, Y_val


def create_dataset_2d(data_type="moons", n_samples=500):
    """
        Erzeugt einen Datensatz mit Punkten in 2D

        Args:
            data_type: String der angibt, was für ein Datensatz erzeugt werden soll
                  (moons, blobs)
            n_samples: Anzahl der erzeugten Datenpunkte

        Returns:
            X: Eingabe, Dimension (2, Anzahl Datenpunke)
            Y: erwartete Ausgabe, Dimension (2, Anzahl Datenpunke)
    """

    if data_type == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, random_state=1, cluster_std=3)
    elif data_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=1)
    else:
        raise AttributeError("Typ der Daten unbekannt!")

    # Daten in richtiges Format bringen
    X = np.transpose(X)
    Y = get_one_hot(y, 2)

    return X, Y


def plot_decision_boundary_2d(model, X, Y, cmap="Spectral"):
    """
        Erzeuge Grafik für Entscheidungsgrenze

        Args:
            model: neuronales Netz
            X: Eingabe, Dimension (2, Anzahl Datenpunke)
            Y: erwartete Ausgabe, Dimension (2, Anzahl Datenpunke)
            cmap: Matplotlib Farbmap
    """

    # finde minimalen und maximalen Wert der Daten und füge Padding hinzu
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    # erzeuge Grid mit Abtstand dist zwischen den Punkten
    dist = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, dist), np.arange(y_min, y_max, dist))

    # Predict the function value for the whole grid
    # Inferenz des neuronalen Netzes und Umwandlung in Gridform
    Z = model.predict(np.transpose(np.c_[xx.ravel(), yy.ravel()]))
    Z = np.argmax(Z, axis=0).reshape(xx.shape)

    # erzeuge Grafik mit Datenpunken und der Entscheidungsgrenz
    plt.contourf(xx, yy, Z, cmap=plt.get_cmap(cmap))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plot_data_2d(X, Y, cmap=plt.get_cmap(cmap))
