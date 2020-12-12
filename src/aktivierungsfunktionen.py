import numpy as np


class Sigmoid:
    """
        Klasse, die die Sigmoid-Aktivierungsfunktion implementiert
    """

    @staticmethod
    def forward(X):
        """ f(X) """
        return 1. / (1. + np.exp(-X))

    @staticmethod
    def backward(X):
        """ f'(X) """
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
        """ f'(X) """
        X[X < 0] = 0
        X[X > 0] = 1
        return X


class Softmax:
    """
        Klasse, die die Softmax-Aktivierungsfunktion implementiert
    """

    @staticmethod
    def forward(X):
        """ f(X) """
        counter = np.exp(X)
        denominator = np.sum(counter, axis=0)
        return np.divide(counter, denominator)

    @staticmethod
    def backward(X):
        """ f'(X) """
        S = Softmax.forward(X)

        ds_list = []
        for i in range(X.shape[1]):
            s = S[:, [i]]
            diagonal = np.diag(s.flatten())
            tmp_matrix = np.tile(s, s.shape[0])
            dx = diagonal - np.multiply(tmp_matrix, np.transpose(tmp_matrix))
            ds_list.append(dx)

        dS = np.stack(tuple(ds_list), axis=0)
        return dS
