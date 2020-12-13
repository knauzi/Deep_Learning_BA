import numpy as np


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
