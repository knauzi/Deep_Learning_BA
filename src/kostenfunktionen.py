import numpy as np


class QK:
    """
        Klasse, die die Quadratische Kostenfunktion implementiert
    """

    @staticmethod
    def compute(AL, Y):
        """
            Berechnung der Kosten

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                Y: Erwarteter Output mit Werten in {0,1},
                   Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)

            Returns:
                cost: Kosten (integer)
        """

        n = Y.shape[1]
        cost = np.nansum(np.square(np.subtract(AL, Y))) / n
        # cost = np.square(np.subtract(AL, Y)).mean()
        cost = np.squeeze(cost)

        return cost

    # @staticmethod
    # def prime(AL, Y):
    #     """
    #         Berechnung der Ableitung der Kostenfunktion nach der Aktivierung der
    #         Output-Schicht
    #
    #         Args:
    #             AL: Aktivierung der Output-Schicht, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
    #             Y: Erwarteter Output mit Werten in {0,1},
    #                Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
    #
    #         Returns:
    #             prime: Ableitung der Kostenfunktion
    #     """
    #
    #     prime = AL - Y
    #     return prime


class BKE:
    """
        Klasse, die die binäre Kreuzentropie Kostenfunktion implementiert
    """

    @staticmethod
    def compute(AL, Y):
        """
            Berechnung der Kosten

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                Y: Erwarteter Output mit Werten in {0,1},
                   Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)

            Returns:
                cost: Kosten (integer)
        """

        n = Y.shape[1]
        cost = - np.nansum(np.multiply(Y, np.log(AL + 1e-8))
                           + np.multiply((1 - Y), np.log(1 - AL + 1e-8))) / n
        # cost = - np.mean(np.multiply(Y, np.log(AL + 1e-10)) + np.multiply((1 - Y), np.log(1 - AL + 1e-10)))
        cost = np.squeeze(cost)

        return cost

    # @staticmethod
    # def prime(AL, Y):
    #     """
    #         Berechnung der Ableitung der Kostenfunktion nach der Aktivierung der
    #         Output-Schicht
    #
    #         Args:
    #             AL: Aktivierung der Output-Schicht, Dimension (1, Anzahl Datenpunkte)
    #             Y: Erwarteter Output mit Werten in {0,1},
    #                Dimension (1, Anzahl Datenpunkte)
    #
    #         Returns:
    #             dAL: Ableitung der Kostenfunktion nach der Aktivierung der Output-
    #                  Schicht
    #     """
    #
    #     dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #
    #     return dAL


class KE:
    """
        Klasse, die die Kreuzentropie Kostenfunktion implementiert
    """

    @staticmethod
    def compute(AL, Y):
        """
            Berechnung der Kosten

            Args:
                AL: Aktivierung der Output-Schicht, Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)
                Y: Erwarteter Output mit Werten in {0,1},
                   Dimension (Anzahl Ausgaben, Anzahl Datenpunkte)

            Returns:
                cost: Kosten (integer)
        """

        n = Y.shape[1]
        cost = - np.nansum(np.multiply(Y, np.log(AL + 1e-8))) / n
        # cost = - np.mean(np.multiply(Y, np.log(AL + 1e-10)))
        cost = np.squeeze(cost)

        return cost

    # @staticmethod
    # def prime(AL, Y):
    #     pass

