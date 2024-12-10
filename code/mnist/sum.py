import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameter
n = 24
alpha = 0.01
p0 = 0.5
delta = 3.6425

def guetefunktion(p):
    # X = 19, 20, ..., 24
    probs = [binom.pmf(k, n, p) for k in range(19, 25)]
    # X = 20, ..., 24
    guete = sum(probs[1:])
    # X = 19
    guete += delta * probs[0]
    return guete

p_values = np.linspace(0, 1, 100)
gamma_values = [guetefunktion(p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, gamma_values, label="Gütefunktion $\\gamma(p)$", color="blue")
plt.axhline(y=alpha, color="red", linestyle="--", label=f"$\\alpha = {alpha}$")
plt.axvline(x=p0, color="green", linestyle="--", label=f"$p_0 = {p0}$")

plt.xlabel("Probability of Success ($p$)")
plt.ylabel("Probability")
plt.title("Gütefunktion des Tests mit $1_{\{X > 19\}} + \delta*1_{\{X = 19\}}$")
plt.legend()
plt.grid(True)
plt.show()
