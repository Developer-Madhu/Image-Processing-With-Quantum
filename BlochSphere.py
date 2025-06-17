import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector, visualize_transition
import matplotlib.pyplot as plt

# Step 1: Define original theta
theta = 2 * np.arccos(np.sqrt(3)/2) #modify Alpha value based on your desired superposition

# Step 2: Build circuit
qc = QuantumCircuit(1)
qc.ry(theta, 0)  # Prepare state: cos(θ/2)|0⟩ + sin(θ/2)|1⟩
qc.x(0)          # Apply X gate  #You can replace X with Y or Z to see the working of  Y&Z gates

# Step 3: Simulate
backend = Aer.get_backend("statevector_simulator")
compiled = transpile(qc, backend)
result = backend.run(compiled).result()
state = result.get_statevector()

# Step 4: Extract amplitudes
alpha_prime = state[0]  # amplitude of |0⟩
beta_prime = state[1]   # amplitude of |1⟩

# Step 5: Compute new theta'
theta_prime = 2 * np.arccos(np.abs(alpha_prime))
theta_prime_deg = np.degrees(theta_prime)

# Step 6: Display angles and plot
print(f"After X gate:")
print(f"Amplitude of |0⟩ (α') = {alpha_prime}")
print(f"Amplitude of |1⟩ (β') = {beta_prime}")
print(f"Theta after X gate (radians): {theta_prime:.4f}")
print(f"Theta after X gate (degrees): {theta_prime_deg:.2f}°")

# Step 7: Plot Bloch vector
plot_bloch_multivector(state)
plt.show()

# Step 8: Show transition animation
visualize_transition(qc)