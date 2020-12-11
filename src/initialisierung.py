import numpy as np


def initialize_parameters(layer_dims):
    """
        Initialisierung der Gewichte mittels Standard-Normalverteilung und der 
        Verzerrungen(Bias) mit Null; Nummerierung: 2, ..., L

        Args:
            layer_dims: Liste mit Dimensionen aller Schichten

        Returns:
            paramters: Python Dictionary mit allen initialsierten Parametern
                       W2, b2, ..., WL, bL
    """

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W'+str(l+1)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.5
        #parameters['b'+str(l+1)] = np.zeros((layer_dims[l], 1))
        parameters['b'+str(l+1)] = np.random.randn(layer_dims[l], 1) * 0.5

    return parameters
