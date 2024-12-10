import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameter
n = 24
alpha = 0.01
p0 = 0.5

# Funktion zur Berechnung der Gütefunktion für 1_{X >= 19}
def guetefunktion(p):
    # Wahrscheinlichkeiten für die Ereignisse X = 19, 20, ..., 24
    return sum(binom.pmf(k, n, p) for k in range(19, 25))

# Wertebereich für p
p_values = np.linspace(0, 1, 100)
gamma_values = [guetefunktion(p) for p in p_values]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(p_values, gamma_values, label="Gütefunktion $\\gamma(p)$", color="blue")
plt.axhline(y=alpha, color="red", linestyle="--", label=f"$\\alpha = {alpha}$")
plt.axvline(x=p0, color="green", linestyle="--", label=f"$p_0 = {p0}$")

# Achsenbeschriftungen und Titel
plt.xlabel("Probability of Success ($p$)")
plt.ylabel("Probability")
plt.title("Gütefunktion des Tests mit $1_{\{X \\geq 19\}}$")
plt.legend()
plt.grid(True)
plt.show()
