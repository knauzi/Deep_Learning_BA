import matplotlib.pyplot as plt
import numpy as np

"""
# Sigmoid Funktion
x = np.linspace(-10,10,500)
y = 1 / (1 + np.exp(-x))
plt.figure(figsize=(8,5))
plt.title("Sigmoid-Funktion", fontsize=15)
plt.grid()
plt.ylabel("$\sigma(z)$", fontsize=15)
plt.xlabel("z", fontsize=15)
plt.plot(x, y, linewidth=3)
plt.show()

# Sigmoid Funktion mit Verschiebung und Variationen
x = np.linspace(-10,10,500)
# y = 1 / (1 + np.exp(-x))
y1 = 1 / (1 + np.exp(-(3*(x-5))))
y2 = 1 / (1 + np.exp(-(-3*(x-5))))
y3 = 1 / (1 + np.exp(-(5*(x+2))))
plt.figure(figsize=(8,5))
plt.title("Sigmoid-Funktion mit Verschiebung und Skalierung", fontsize=15)
plt.grid()
plt.ylabel("$\sigma(z)$", fontsize=15)
plt.xlabel("z", fontsize=15)
plt.plot(x, y1, linewidth=3)
plt.plot(x, y2, linewidth=3)
plt.plot(x, y3, linewidth=3)
plt.legend(("$\sigma(3(z-5))$", "$\sigma(-3(z-5))$", "$\sigma(5(z+2))$"), loc=6, fontsize=15)
plt.show()
"""

# ReLU
x = np.linspace(-10,10,500)
y = np.where(x < 0, 0, x)
plt.figure(figsize=(8,5))
plt.title("ReLU", fontsize=15)
plt.grid()
plt.ylabel("$g(z)$", fontsize=15)
plt.xlabel("z", fontsize=15)
plt.plot(x, y, linewidth=3)
plt.show()
