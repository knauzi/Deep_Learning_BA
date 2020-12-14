import numpy as np


def init_random_normal(layer_dims):
    """
        Initialisierung der Gewichte und Verzerrungen mittels skalierter Standard-Normalverteilung;
        Nummerierung: 2,...,L

        Args:
            layer_dims: Liste mit Dimensionen aller Schichten

        Returns:
            paramters: Python Dictionary mit allen initialsierten Parametern
                       W2, b2, ..., WL, bL
    """

    parameters = {}
    L = len(layer_dims)
    scale = 0.5

    for l in range(1, L):
        parameters['W'+str(l+1)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
        parameters['b'+str(l+1)] = np.random.randn(layer_dims[l], 1) * scale

    return parameters


def init_xavier_uniform(layer_dims, sigmoid=True):
    """
        Initialisierung der Gewichte mittels Xavier-Initialisierung;
        Nummerierung: 2,...,L

        Args:
            layer_dims: Liste mit Dimensionen aller Schichten
            sigmoid: Boolean der angibt, ob die Initialisierung für die Sigmoid-Aktivierungsfunktion
                     erfolgt (=> mit Faktor 4 multiplizieren)

        Returns:
            paramters: Python Dictionary mit allen initialsierten Parametern
                       W2, b2, ..., WL, bL
    """

    parameters = {}
    L = len(layer_dims)
    scale = 4 if sigmoid else 1

    for l in range(1, L):
        limit = np.sqrt(6. / (layer_dims[l-1] + layer_dims[l]))
        parameters['W'+str(l+1)] = np.random.uniform(-limit, limit, (layer_dims[l], layer_dims[l-1])) \
                                   * scale
        parameters['b'+str(l+1)] = np.zeros((layer_dims[l], 1))

    return parameters


def init_he_uniform(layer_dims):
    """
        Initialisierung der Gewichte und Verzerrungen mittels He-Initialisierung;
        Nummerierung: 2,...,L

        Args:
            layer_dims: Liste mit Dimensionen aller Schichten

        Returns:
            paramters: Python Dictionary mit allen initialsierten Parametern
                       W2, b2, ..., WL, bL
    """

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        limit = np.sqrt(6. / layer_dims[l])
        parameters['W' + str(l + 1)] = np.random.uniform(-limit, limit, (layer_dims[l], layer_dims[l - 1]))
        parameters['b' + str(l + 1)] = np.zeros((layer_dims[l], 1))

    return parameters


def init_zeros(layer_dims):
    """
        Initialisierung der Gewichte und Verzerrungen mit Null
        Nummerierung: 2,...,L

        Args:
            layer_dims: Liste mit Dimensionen aller Schichten

        Returns:
            paramters: Python Dictionary mit allen initialsierten Parametern
                       W2, b2, ..., WL, bL
    """

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l + 1)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l + 1)] = np.zeros((layer_dims[l], 1))

    return parameters
