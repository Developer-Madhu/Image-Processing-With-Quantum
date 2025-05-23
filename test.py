from qiskit.visualization import plot_bloch_vector
from numpy import pi, cos, sin
import matplotlib.pyplot as plt

# Define Bloch vector for a qubit state with θ = π/2, φ = π/2
theta = pi / 2
phi = pi / 2

bloch_vector = [
    cos(phi) * sin(theta),  # x
    sin(phi) * sin(theta),  # y
    cos(theta)              # z
]

# Plot Bloch sphere
bloch_plot = plot_bloch_vector(bloch_vector)
bloch_plot.savefig("bloch_sphere.png")  # Saves the plot as an image
bloch_plot.show()
