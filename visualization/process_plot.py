import numpy as np
import matplotlib.pyplot as plt

h = 5
O = 3
t = 0.35

m = O * np.exp(-h * t)
v = (1 - np.exp(-2 * h * t)) / (2 * h)

print("m = ", m)
print("v = ", v)

def l(x):
    return np.log(x / (1 - x))

def s(x):
    return 1 / (1 + np.exp(-x))

def p(x):
    return (1 / (np.sqrt(2 * np.pi * v) * x * (1 - x))) * np.exp(-np.power((l(x) - m), 2) / (2 * v))

def d(x):
    return (l(x) - 2 * v * x - m + v) / (v * x * (x - 1))

def d2(x):
    return (x * (x - 1) / v) * (l(x) - 2 * v * x - m + v)

print("p(0.5) = ", p(0.5))
print("d(0.5) = ", d(0.5))
print("d2(0.0014) = ", d2(0.0014))

import seaborn as sns

# Set the theme to whitegrid and customize the style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2})

colors = ["#FF9B85", "#73A88C", "#6991CF"]
# Set the font to be LaTeX-like
# plt.rcParams["font.family"] = "serif"
# If you have LaTeX installed on your system, uncomment the following line
plt.rcParams["text.usetex"] = True

def logistic(y):
    return 1 / (1 + np.exp(-y))

y_values = np.linspace(-7, 7, 1000000)  # Generate more evenly spaced points
x_values = logistic(y_values)  # Transform the y values into the interval (0, 1)

plt.figure(figsize=(6, 6))

plt.plot(x_values, p(x_values), color=colors[0], label="$p(x)$")
plt.plot(x_values, d(x_values), color=colors[1], label=r"$\nabla\log p(x)$")
plt.plot(x_values, d2(x_values), color=colors[2], label=r"$g^2(x)\nabla\log p(x)$")

# add vertical dashed line around 0.6
plt.axvline(x=0.6, linestyle='--', color='black', linewidth=1.5)

plt.ylim(-2, 10)  # Limit the range of the y-axis

plt.grid(True)  # Add grid lines
# plt.legend(loc="best")
# Add legend to top left
plt.legend(loc=2, bbox_to_anchor=(0.02, 0.99), borderaxespad=0., prop={'size': 16})
plt.xlabel("x")
sns.despine()  # Remove the top and right spines
plt.tight_layout()  # Make the plot fill out the figure area

plt.show()




