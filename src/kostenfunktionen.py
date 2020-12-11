import numpy as np


class MSE:
    """
        Klasse, die die Quadratische Kostenfunktion implementiert
    """

    @staticmethod
    def compute(AL, Y):
        """
            Berechnung der Kosten

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (1, Anzahl Datenpunkte)
                Y: Erwarteter Output mit der Werten in {0,1},
                   Dimension (1, Anzahl Datenpunkte)

            Returns:
                cost: Kosten (integer)
        """

        cost = np.square(np.subtract(AL, Y)).mean()
        return cost

    @staticmethod
    def prime(AL, Y):
        """
            Berechnung der Ableitung der Kostenfunktion nach der Aktivierung der
            Output-Schicht

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (1, Anzahl Datenpunkte
                Y: Erwarteter Output mit der Werten in {0,1},
                   Dimension (1, Anzahl Datenpunkte)

            Returns:
                prime: Ableitung der Kostenfunktion
        """

        prime = AL - Y
        return prime


# TODO passt noch nicht in die Implementierung der Backpropagation
class BCE:
    """
        Klasse, die die Binary-Cross-Entropy Kostenfunktion implementiert
    """

    @staticmethod
    def compute(AL, Y):
        """
            Berechnung der Kosten

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (1, Anzahl Datenpunkte)
                Y: Erwarteter Output mit der Werten in {0,1},
                   Dimension (1, Anzahl Datenpunkte)

            Returns:
                cost: Kosten (integer)
        """

        m = Y.shape[1]
        cost = -1.0 / m * (np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), 
                           np.log(1 - AL))).sum()
        cost = np.squeeze(cost) 

        return cost

    @staticmethod
    def prime(AL, Y):
        """
            Berechnung der Ableitung der Kostenfunktion nach der Aktivierung der
            Output-Schicht

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (1, Anzahl Datenpunkte)
                Y: Erwarteter Output mit der Werten in {0,1},
                   Dimension (1, Anzahl Datenpunkte)

            Returns:
                dAL: Ableitung der Kostenfunktion nach der Aktivierung der Output-
                     Schicht
        """

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        return dAL
