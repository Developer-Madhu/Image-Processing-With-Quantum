from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time

# Custom complex amplitudes
alpha =np.sqrt(3)/2
beta =complex(0.25, np.sqrt(3)/4)  # β = 0.25 + 0.433i

# Normalize the state
norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
alpha /= norm
beta /= norm

# Function to compute Bloch sphere coordinates and θ (in degrees)
def bloch_coords_and_theta_deg(state):
    a, b = state.data
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a)**2 - abs(b)**2
    theta_rad = np.arccos(z)
    theta_deg = np.degrees(theta_rad)
    return round(x, 3), round(y, 3), round(z, 3), round(theta_deg, 2)

# Display Bloch sphere animation with θ from Z-axis
for theta in np.linspace(0, np.pi/3, 20): # change thetha here
    qc = QuantumCircuit(1)
    qc.initialize([alpha, beta], 0)
    qc.ry(theta, 0)

    state = Statevector.from_instruction(qc)
    x, y, z, theta_deg = bloch_coords_and_theta_deg(state)

    fig = plot_bloch_multivector(state)
    clear_output(wait=True)
    display(fig)
    print(f"Rx(θ = {round(theta, 2)} rad) → x={x}, y={y}, z={z}, θ (from Z-axis) = {theta_deg}°")
    plt.close(fig)
    time.sleep(0.3)