import numpy as np


class Sigmoid:
    """
        Klasse, die die Sigmoid-Aktivierungsfunktion implementiert
    """

    @staticmethod
    def forward(X):
        """ f(x) """
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def backward(X):
        """ f'(x) """
        return Sigmoid.forward(X) * (1 - Sigmoid.forward(X))


class Relu:
    """
        Klasse, die die Relu-Aktivierungsfunktion implementiert
    """

    @staticmethod
    def forward(X):
        """ f(X) """
        X[X < 0] = 0
        return X

    @staticmethod
    def backward(X):
        """ f'(x) """
        X[X < 0] = 0
        X[X > 0] = 1
        return X
