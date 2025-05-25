import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator # Keep for local testing if needed, or remove
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RYGate
import matplotlib.pyplot as plt

# Import for IBM Quantum Runtime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.options import options # For general options if needed
from qiskit_ibm_runtime.options import SamplerOptions # Specific options for Sampler
from qiskit_ibm_runtime import Sampler # Import Sampler primitive

# --- Helper Functions (unchanged from your original code) ---

def hadamard(circ, n):
    """Applies Hadamard gate to a list of qubits."""
    for i in n:
        circ.h(i)

def change(state, new_state):
    """
    Compares two binary states and returns indices where they differ.
    Used to determine which qubits need to be flipped.
    """
    n = len(state)
    c = np.array([])
    for i in range(n):
        if state[i] != new_state[i]:
            c = np.append(c, int(i))
    return c.astype(int) if len(c) > 0 else c

def binary(circ, state, new_state, k):
    """
    Applies X gates to flip qubits based on differences between two binary states.
    (Note: This function is defined but not directly used in your main loop's FRQI logic,
    as `frqi_pixel` handles the position setting.)
    """
    c = change(state, new_state)
    if len(c) > 0:
        for i in c:
            circ.x(i)  # Correctly flip position qubits
    else:
        pass

def cnri(circ, control_qubits, target_qubit, theta):
    """
    Applies a controlled Ry gate (CRy) with multiple controls.
    This is a core component for encoding pixel intensity.
    """
    controls = len(control_qubits)
    cry = RYGate(theta).control(controls)  # Use theta for the rotation angle
    aux = np.append(control_qubits, target_qubit).tolist() # Combine control and target qubits
    circ.append(cry, aux)

def frqi_pixel(circ, control_qubits, target_qubit, angle, position_idx, num_qubits):
    """
    Encodes a single pixel's intensity and performs negation based on FRQI.

    Args:
        circ (QuantumCircuit): The quantum circuit to operate on.
        control_qubits (list): List of qubit indices for position encoding.
        target_qubit (int): Index of the intensity qubit.
        angle (float): The rotation angle (intensity) for the pixel.
        position_idx (int): The decimal index representing the pixel's position.
        num_qubits (int): The number of qubits used for position encoding.
    """
    # Set position qubits to |position_idx>
    # Convert decimal position_idx to a binary string of length num_qubits
    bin_idx = format(position_idx, f'0{num_qubits}b')
    for bit_val, qubit_idx in zip(bin_idx, control_qubits):
        if bit_val == '1':
            circ.x(qubit_idx) # Apply X gate if the corresponding bit is 1

    # Encode intensity with controlled Ry(theta)
    if angle > 0:
        cnri(circ, control_qubits, target_qubit, angle)
    
    circ.barrier(label="After_Encoding")
    
    # Negation: Reset intensity to |0> and apply controlled Ry(pi/2 - theta)
    negated_theta = np.pi/2 - angle
    
    # If the original angle was > 0, the intensity qubit was rotated.
    # To negate, we first "undo" the rotation to bring it back to |0> if it was |1>,
    # then apply the new rotation.
    if angle > 0:
        circ.x(target_qubit) # Flip to bring |1> to |0> for the negation rotation
    
    if negated_theta >= 0: # Ensure angle is non-negative
        cnri(circ, control_qubits, target_qubit, negated_theta)
    
    if angle > 0:
        circ.x(target_qubit) # Restore the basis if it was flipped for negation
    
    circ.barrier(label="After_Negation")

# --- Image Setup ---

# Input 2x2 binary image
image = np.array([
    [1, 0],
    [1, 0]
])
print("\nOriginal Image (Binary):")
print(image)

n = image.shape[0]  # Assuming square image (n=2 for 2x2)
num_pixels = n * n  # Total number of pixels (4 for 2x2)
# Qubits for positions: ceil(log2(num_pixels))
# For 4 pixels, log2(4) = 2, so q = 2 control qubits
q = int(np.ceil(np.log2(num_pixels)))
print(f"Image size: {n}x{n}, using {q} control qubits")

# Compute angles for binary pixels (0 or 1)
# Binary 0 maps to angle 0, Binary 1 maps to angle pi/2
angles = np.pi/2 * image.flatten()

# Compute classical negated image for comparison
negated_image = 1 - image
print("\nNegated Image (Binary):")
print(negated_image)

# Process each pixel by defining its value, position index, and (row, col)
pixels = [
    (image[0,0], 0, (0,0)),  # Pixel at (0,0) -> position index 0 -> |00>
    (image[0,1], 1, (0,1)),  # Pixel at (0,1) -> position index 1 -> |01>
    (image[1,0], 2, (1,0)),  # Pixel at (1,0) -> position index 2 -> |10>
    (image[1,1], 3, (1,1))   # Pixel at (1,1) -> position index 3 -> |11>
]

# --- IBM Quantum Runtime Setup ---

# 1. Load your IBM Quantum account.
# If you haven't saved your account yet, uncomment and run the line below once:
# from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_API_TOKEN', overwrite=True)
# Replace 'YOUR_API_TOKEN' with your actual token from quantum.ibm.com
# The 'ibm_quantum' channel is deprecated; 'ibm_cloud' is the current recommendation.
service = QiskitRuntimeService(
    token='88b7e01d0d72ecec68894a80350a71d8fe48257f19c7304a707a40e35c081a78f52e9b82de45041ea983b973cfd03de6dfe027c12f11a29e264b8f12644c5ab3' # ** IMPORTANT: Replace with your actual IBM Quantum API token **
)

# 2. Choose a backend.
# For real hardware, you would typically use:
# backend = service.least_busy(operational=True, simulator=False)
# Or specify a specific backend name:
# backend = service.get_backend('ibm_osaka')

# For cloud simulator (recommended for testing and statevector access):
backend = service.backend('ibm_brisbane')

# 3. Configure Sampler options.
# Use SamplerOptions for Sampler primitive specific configurations.
options = SamplerOptions()
# Removed options.resilience_level and options.optimization_level
# as they are not directly supported by SamplerOptions.
options.execution.shots = 999999 # Number of measurement shots for statistical results

# 4. Initialize the Sampler primitive.
sampler = Sampler(backend=backend, options=options)

# --- Pixel Processing Loop ---

for pixel_value, pos_idx, pos in pixels:
    # Quantum Circuit Setup for each pixel
    # q + 1 qubits: 'q' for position, +1 for intensity qubit
    qr = QuantumRegister(q + 1, 'q')
    cr = ClassicalRegister(1, 'c') # Classical register to measure the intensity qubit
    qc = QuantumCircuit(qr, cr)
    
    control_qubits = list(range(q)) # Qubits 0 to q-1 are control qubits for position
    target_qubit = q # The last qubit is the intensity qubit
    
    # Build FRQI circuit for this pixel
    angle = angles[pos_idx] # Get the intensity angle for the current pixel
    frqi_pixel(qc, control_qubits, target_qubit, angle, pos_idx, q)
    
    # Measure the intensity qubit to get the negated pixel value
    qc.measure(target_qubit, cr)
    
    # Print the quantum circuit diagram
    print(f"\nFRQI Circuit for Pixel {pixel_value} at Position {pos}:")
    print(qc.draw(output="text"))
    
    # Run the circuit using the Sampler primitive
    job = sampler.run(qc)
    result = job.result() # Wait for the job to complete and get results
    
    # --- Handling Statevector Information ---
    # Note: When running on a Sampler primitive (especially a cloud one),
    # you typically get measurement outcomes (counts/probabilities), not direct statevectors.
    # To get statevectors, you would run on a statevector simulator backend *without* measurements
    # and use `qc.save_statevector()`.
    print("\nStatevector BEFORE negation: (Not directly available from Runtime Primitives with measurements)")
    print("Statevector AFTER negation: (Not directly available from Runtime Primitives with measurements)")
    print("To get statevectors, run on 'simulator_statevector' backend *without* measurements and use qc.save_statevector()")

    # --- Estimate Negated Pixel Value from Measurement Counts ---
    # The Sampler returns quasi_dists, which are probability distributions.
    # For a single measured qubit, '0' or '1' are the possible outcomes.
    counts_prob = result.quasi_dists[0].binary_probabilities()
    
    # Convert probabilities back to "counts" for your original calculation logic
    total_shots = options.execution.shots
    count_1 = counts_prob.get('1', 0) * total_shots
    count_0 = counts_prob.get('0', 0) * total_shots
    
    total = count_0 + count_1
    negated_pixel = 0
    if total > 0:
        prob_1 = count_1 / total
        # Reconstruct the angle from the probability of |1>
        theta_prime = 2 * np.arcsin(np.sqrt(prob_1))
        # Convert the angle back to a binary pixel value (0 or 1)
        negated_pixel = int((theta_prime / (np.pi/2)) + 0.5) # Round to nearest 0 or 1
    
    print(f"\nExpected Negated Pixel Value: {1 - pixel_value}")
    print(f"Measured Negated Pixel Value: {negated_pixel}")

# --- Plotting Negated Image ---
# This part uses the classically negated image for plotting, as the quantum
# process is pixel-by-pixel and doesn't directly reconstruct an image.
plt.figure(figsize=(4, 2))
plt.imshow(negated_image, cmap='gray', vmin=0, vmax=1)
plt.title("Negated Image (Quantum)")
plt.axis('off')
plt.savefig('negated_image_binary.png')
plt.close()
