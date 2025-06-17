# import numpy as np
# import matplotlib.pyplot as plt
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# from qiskit.circuit.library import MCXGate

# original = np.array([[0, 1], [2, 3]])

# def classical_nearest_neighbor_upscale(img):
#     """Classical implementation for comparison"""
#     rows, cols = img.shape
#     upscale = np.zeros((rows * 2, cols * 2), dtype=int)
#     for i in range(rows):
#         for j in range(cols):
#             val = img[i, j]
#             upscale[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = val
#     return upscale

# def quantum_nearest_neighbor_upscale(img):
#     """
#     Quantum implementation of nearest neighbor upscaling.
#     Creates a quantum circuit that maps original image positions to upscaled positions
#     and replicates pixel values according to nearest neighbor logic.
#     """
#     rows, cols = img.shape
#     original_size = rows * cols
#     upscaled_size = (rows * 2) * (cols * 2)
    
#     # Calculate required qubits
#     original_pos_qubits = int(np.ceil(np.log2(original_size)))
#     upscaled_pos_qubits = int(np.ceil(np.log2(upscaled_size)))
#     gray_qubits = 4  # For grayscale values 0-15
    
#     # Create quantum registers
#     orig_pos_reg = QuantumRegister(original_pos_qubits, name='orig_pos')
#     upscaled_pos_reg = QuantumRegister(upscaled_pos_qubits, name='upscaled_pos')
#     gray_reg = QuantumRegister(gray_qubits, name='gray')
    
#     # Classical registers for measurement
#     c_orig_pos = ClassicalRegister(original_pos_qubits, name='c_orig_pos')
#     c_upscaled_pos = ClassicalRegister(upscaled_pos_qubits, name='c_upscaled_pos')
#     c_gray = ClassicalRegister(gray_qubits, name='c_gray')
    
#     qc = QuantumCircuit(orig_pos_reg, upscaled_pos_reg, gray_reg, 
#                        c_orig_pos, c_upscaled_pos, c_gray)
    
#     # Step 1: Create superposition of original image positions
#     for qubit in orig_pos_reg:
#         qc.h(qubit)
    
#     # Step 2: Quantum nearest neighbor mapping
#     # For each original position, map to corresponding upscaled positions
#     def add_quantum_upscaling_logic():
#         for orig_i in range(rows):
#             for orig_j in range(cols):
#                 orig_linear_idx = orig_i * cols + orig_j
#                 pixel_value = img[orig_i, orig_j]
                
#                 # Each original pixel maps to 4 upscaled pixels (2x2 block)
#                 upscaled_positions = []
#                 for di in range(2):  # 2x2 replication
#                     for dj in range(2):
#                         upscaled_i = orig_i * 2 + di
#                         upscaled_j = orig_j * 2 + dj
#                         upscaled_linear_idx = upscaled_i * (cols * 2) + upscaled_j
#                         upscaled_positions.append(upscaled_linear_idx)
                
#                 # Create quantum controls for this original position
#                 orig_binary = f"{orig_linear_idx:0{original_pos_qubits}b}"
                
#                 # Apply controls for original position
#                 control_qubits = []
#                 for bit_idx, bit in enumerate(orig_binary):
#                     if bit == '0':
#                         qc.x(orig_pos_reg[bit_idx])
#                         control_qubits.append(orig_pos_reg[bit_idx])
#                     else:
#                         control_qubits.append(orig_pos_reg[bit_idx])
                
#                 # For each of the 4 upscaled positions this original pixel maps to
#                 for upscaled_idx in upscaled_positions:
#                     upscaled_binary = f"{upscaled_idx:0{upscaled_pos_qubits}b}"
                    
#                     # Set upscaled position qubits
#                     for bit_idx, bit in enumerate(upscaled_binary):
#                         if bit == '1':
#                             if len(control_qubits) == 1:
#                                 qc.cx(control_qubits[0], upscaled_pos_reg[bit_idx])
#                             else:
#                                 qc.mcx(control_qubits, upscaled_pos_reg[bit_idx])
                    
#                     # Set grayscale value
#                     pixel_binary = f"{pixel_value:04b}"
#                     for bit_idx, bit in enumerate(pixel_binary):
#                         if bit == '1':
#                             if len(control_qubits) == 1:
#                                 qc.cx(control_qubits[0], gray_reg[bit_idx])
#                             else:
#                                 qc.mcx(control_qubits, gray_reg[bit_idx])
                    
#                     qc.barrier()  # Separate different upscaled positions
                
#                 # Reset control bits
#                 for bit_idx, bit in enumerate(orig_binary):
#                     if bit == '0':
#                         qc.x(orig_pos_reg[bit_idx])
                
#                 qc.barrier()  # Separate different original positions
    
#     add_quantum_upscaling_logic()
    
#     # Step 3: Measure all registers
#     qc.measure(orig_pos_reg, c_orig_pos)
#     qc.measure(upscaled_pos_reg, c_upscaled_pos)
#     qc.measure(gray_reg, c_gray)
    
#     return qc

# def create_quantum_image_circuit(matrix, name):
#     """Create a quantum circuit for image representation"""
#     rows, cols = matrix.shape
#     size = rows * cols
#     pos_qubits = int(np.ceil(np.log2(size)))
#     gray_qubits = 4
    
#     pos_reg = QuantumRegister(pos_qubits, name=f'{name}_pos')
#     gray_reg = QuantumRegister(gray_qubits, name=f'{name}_gray')
#     c_pos = ClassicalRegister(pos_qubits, name=f'c_{name}_pos')
#     c_gray = ClassicalRegister(gray_qubits, name=f'c_{name}_gray')
    
#     qc = QuantumCircuit(pos_reg, gray_reg, c_pos, c_gray)
    
#     # Create superposition of positions
#     for qubit in pos_reg:
#         qc.h(qubit)
    
#     # Encode pixel values
#     for i in range(rows):
#         for j in range(cols):
#             linear_idx = i * cols + j
#             pixel_value = matrix[i, j]
            
#             # Create position controls
#             pos_binary = f"{linear_idx:0{pos_qubits}b}"
#             control_qubits = []
            
#             for bit_idx, bit in enumerate(pos_binary):
#                 if bit == '0':
#                     qc.x(pos_reg[bit_idx])
#                     control_qubits.append(pos_reg[bit_idx])
#                 else:
#                     control_qubits.append(pos_reg[bit_idx])
            
#             # Set grayscale value
#             pixel_binary = f"{pixel_value:04b}"
#             for bit_idx, bit in enumerate(pixel_binary):
#                 if bit == '1':
#                     if len(control_qubits) == 1:
#                         qc.cx(control_qubits[0], gray_reg[bit_idx])
#                     else:
#                         qc.mcx(control_qubits, gray_reg[bit_idx])
            
#             # Reset position controls
#             for bit_idx, bit in enumerate(pos_binary):
#                 if bit == '0':
#                     qc.x(pos_reg[bit_idx])
    
#     qc.measure(pos_reg, c_pos)
#     qc.measure(gray_reg, c_gray)
    
#     return qc

# # Generate upscaled image using both methods
# classical_upscaled = classical_nearest_neighbor_upscale(original)
# quantum_upscaling_circuit = quantum_nearest_neighbor_upscale(original)

# print(f"Original matrix:\n{original}")
# print(f"Original matrix shape: {original.shape}")

# print(f"\nClassical upscaled matrix:\n{classical_upscaled}")
# print(f"Classical upscaled matrix shape: {classical_upscaled.shape}")

# print(f"\nQuantum upscaling circuit created!")
# print(f"Circuit qubits: {quantum_upscaling_circuit.num_qubits}")
# print(f"Circuit depth: {quantum_upscaling_circuit.depth()}")
# print(f"Circuit operations: {len(quantum_upscaling_circuit.data)}")

# # Create individual circuits for comparison
# original_circuit = create_quantum_image_circuit(original, "original")
# upscaled_circuit = create_quantum_image_circuit(classical_upscaled, "upscaled")

# print(f"\nOriginal image circuit: {original_circuit.num_qubits} qubits, depth {original_circuit.depth()}")
# print(f"Upscaled image circuit: {upscaled_circuit.num_qubits} qubits, depth {upscaled_circuit.depth()}")

# def visualize_quantum_circuits():
#     """Visualize the quantum circuits"""
#     try:
#         fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
#         style = {
#             'backgroundcolor': '#ffffff',
#             'textcolor': '#000000',
#             'gatefacecolor': '#ffffff',
#             'gateedgecolor': '#000000',
#             'cregbundle': True,
#             'fold': -1
#         }
        
#         # Original image circuit
#         original_circuit.draw('mpl', ax=axes[0,0], style=style)
#         axes[0,0].set_title('Original Image (2×2)\nQuantum Circuit', fontweight='bold')
        
#         # Upscaled image circuit
#         upscaled_circuit.draw('mpl', ax=axes[0,1], style=style)
#         axes[0,1].set_title('Upscaled Image (4×4)\nQuantum Circuit', fontweight='bold')
        
#         # Quantum upscaling circuit (simplified view)
#         try:
#             # Create a simplified version for visualization
#             simplified_qc = QuantumCircuit(3, 3)  # Simplified for display
#             simplified_qc.h(0)
#             simplified_qc.h(1)
#             simplified_qc.barrier()
#             simplified_qc.cx(0, 2)
#             simplified_qc.cx(1, 2)
#             simplified_qc.barrier()
#             simplified_qc.measure_all()
            
#             simplified_qc.draw('mpl', ax=axes[1,0], style=style)
#             axes[1,0].set_title('Quantum Nearest Neighbor\nUpscaling Logic (Simplified)', fontweight='bold')
#         except:
#             axes[1,0].text(0.5, 0.5, 'Quantum Upscaling Circuit\n(Too complex to display)', 
#                           ha='center', va='center', transform=axes[1,0].transAxes)
#             axes[1,0].set_title('Quantum Upscaling Circuit', fontweight='bold')
        
#         # Matrix comparison
#         axes[1,1].axis('off')
        
#         # Display matrices
#         axes[1,1].text(0.25, 0.8, 'Original Matrix:', ha='center', fontweight='bold', 
#                       transform=axes[1,1].transAxes)
#         axes[1,1].text(0.25, 0.6, str(original), ha='center', fontfamily='monospace',
#                       transform=axes[1,1].transAxes)
        
#         axes[1,1].text(0.25, 0.4, '↓ Quantum Upscaling ↓', ha='center', fontweight='bold',
#                       transform=axes[1,1].transAxes)
        
#         axes[1,1].text(0.75, 0.8, 'Upscaled Matrix:', ha='center', fontweight='bold',
#                       transform=axes[1,1].transAxes)
#         axes[1,1].text(0.75, 0.5, str(classical_upscaled), ha='center', fontfamily='monospace',
#                       transform=axes[1,1].transAxes)
        
#         plt.suptitle('Quantum Nearest Neighbor Image Upscaling', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig('quantum_nearest_neighbor_upscaling.png', dpi=300, bbox_inches='tight')
#         print("Visualization saved as 'quantum_nearest_neighbor_upscaling.png'")
#         plt.show()
        
#         return True
        
#     except Exception as e:
#         print(f"Visualization error: {e}")
#         return False

# print("\n--- Visualizing Quantum Circuits ---")
# viz_success = visualize_quantum_circuits()

# if not viz_success:
#     print("Visualization failed. Showing text representation...")
#     try:
#         print("\nOriginal circuit (text):")
#         print(original_circuit.draw(output='text', fold=80))
#         print(f"\nQuantum upscaling circuit has {quantum_upscaling_circuit.num_qubits} qubits")
#         print("(Too complex for text display)")
#     except Exception as e:
#         print(f"Text representation failed: {e}")

# print("\n" + "="*80)
# print("QUANTUM NEAREST NEIGHBOR UPSCALING SUMMARY")
# print("="*80)
# print(f"Original matrix: {original.shape} → {original.size} pixels")
# print(f"Upscaled matrix: {classical_upscaled.shape} → {classical_upscaled.size} pixels")
# print(f"Upscaling factor: 2x in each dimension (4x total pixels)")
# print()
# print("QUANTUM IMPLEMENTATION FEATURES:")
# print("• Quantum superposition over original image positions")
# print("• Quantum-controlled replication of pixels to 2x2 blocks")
# print("• Each original pixel quantumly maps to 4 upscaled positions")
# print("• Maintains nearest neighbor interpolation logic in quantum domain")
# print()
# print(f"Quantum upscaling circuit: {quantum_upscaling_circuit.num_qubits} qubits")
# print(f"Circuit depth: {quantum_upscaling_circuit.depth()}")
# print(f"Total quantum operations: {len(quantum_upscaling_circuit.data)}")
# print()
# print("The quantum circuit implements the same nearest neighbor logic")
# print("as the classical algorithm but using quantum superposition and")
# print("controlled operations to process all pixel mappings simultaneously.")
# print("="*80)
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import Aer
import matplotlib.pyplot as plt

# Input 2x2 matrix
original = np.array([[0, 1], [2, 3]])

# Define parameters based on the original matrix
rows_orig, cols_orig = original.shape
size_orig = rows_orig * cols_orig
pos_qubits_orig = int(np.ceil(np.log2(size_orig)))
gray_qubits = 4 # Grayscale bits are consistent

# Define parameters for the upscaled matrix
rows_upscaled, cols_upscaled = rows_orig * 2, cols_orig * 2
size_upscaled = rows_upscaled * cols_upscaled
pos_qubits_upscaled = int(np.ceil(np.log2(size_upscaled)))


# --- Modified Functions to operate on a pre-defined QuantumCircuit ---

def encode_quantum_image(qc, matrix, pos_reg, gray_reg):
    rows, cols = matrix.shape
    pos_qubits = pos_reg.size # Get size from the register object

    # Superposition for all positions
    for qubit in pos_reg:
        qc.h(qubit)

    # Encode grayscale values
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            pixel_val = matrix[i, j]
            bin_pos = f"{idx:0{pos_qubits}b}"
            ctrl_qubits = []

            for b, bit in enumerate(bin_pos):
                if bit == '0':
                    qc.x(pos_reg[b])
                ctrl_qubits.append(pos_reg[b])

            pixel_bin = f"{pixel_val:04b}"
            for k, bit in enumerate(pixel_bin):
                if bit == '1':
                    if len(ctrl_qubits) == 1:
                        qc.cx(ctrl_qubits[0], gray_reg[k])
                    else:
                        qc.mcx(ctrl_qubits, gray_reg[k])

            for b, bit in enumerate(bin_pos):
                if bit == '0':
                    qc.x(pos_reg[b])

def apply_quantum_nearest_neighbor_upscale(qc, img, orig_pos_reg, up_pos_reg, gray_reg):
    rows, cols = img.shape
    orig_pos_qubits = orig_pos_reg.size
    upscaled_pos_qubits = up_pos_reg.size

    # Superposition for original positions (already done in encode_quantum_image, but for clarity if this were separate)
    # for q in orig_pos_reg:
    #     qc.h(q) # This line should ideally only be in the initial setup

    # Note: The original code applies H to orig_pos in upscaling function again.
    # If the idea is to prepare the state for upscaling from an already prepared
    # original image state, then the H gates for orig_pos_reg should only be applied once.
    # Assuming `encode_quantum_image` already handles superposition of `orig_pos_reg`.

    for i in range(rows):
        for j in range(cols):
            orig_idx = i * cols + j
            pixel_val = img[i, j]
            upscaled_idxs = []

            for di in range(2):
                for dj in range(2):
                    ui = i * 2 + di
                    uj = j * 2 + dj
                    upscaled_idxs.append(ui * (cols * 2) + uj)

            ctrl_bits = f"{orig_idx:0{orig_pos_qubits}b}"
            controls = []
            for b, bit in enumerate(ctrl_bits):
                if bit == '0':
                    qc.x(orig_pos_reg[b])
                controls.append(orig_pos_reg[b])

            for up_idx in upscaled_idxs:
                up_bits = f"{up_idx:0{upscaled_pos_qubits}b}"
                for b, bit in enumerate(up_bits):
                    if bit == '1':
                        # The controls are from the original_position register.
                        # We apply a multi-controlled X gate to set the corresponding
                        # bit in the *upscaled* position register if the original position is active.
                        if len(controls) == 1:
                            qc.cx(controls[0], up_pos_reg[b])
                        else:
                            qc.mcx(controls, up_pos_reg[b])

                pixel_bin = f"{pixel_val:04b}"
                for b, bit in enumerate(pixel_bin):
                    if bit == '1':
                        # Similarly, apply the pixel value to the gray_reg
                        # when the original position is active.
                        if len(controls) == 1:
                            qc.cx(controls[0], gray_reg[b])
                        else:
                            qc.mcx(controls, gray_reg[b])

                qc.barrier() # Barrier after each upscaled pixel encoding (optional, for visualization)

            # Uncompute the X gates applied to controls for this iteration
            for b, bit in enumerate(ctrl_bits):
                if bit == '0':
                    qc.x(orig_pos_reg[b])
            qc.barrier() # Barrier after uncomputing controls for each original pixel (optional)


# --- Main execution flow ---

# 1. Define all necessary quantum and classical registers for the full circuit
orig_pos_reg = QuantumRegister(pos_qubits_orig, name='orig_pos')
upscaled_pos_reg = QuantumRegister(pos_qubits_upscaled, name='up_pos')
gray_reg = QuantumRegister(gray_qubits, name='gray') # Grayscale is shared

c_orig_pos = ClassicalRegister(pos_qubits_orig, name='c_orig_pos')
c_up_pos = ClassicalRegister(pos_qubits_upscaled, name='c_up_pos')
c_gray = ClassicalRegister(gray_qubits, name='c_gray')

# 2. Create the single, master QuantumCircuit
full_qc = QuantumCircuit(orig_pos_reg, upscaled_pos_reg, gray_reg,
                         c_orig_pos, c_up_pos, c_gray)

# 3. Apply the "input image" encoding logic to the full_qc
encode_quantum_image(full_qc, original, orig_pos_reg, gray_reg)

full_qc.barrier(label='Input Image Encoded') # Add a barrier for clarity

# 4. Apply the "upscaling" logic to the same full_qc
apply_quantum_nearest_neighbor_upscale(full_qc, original, orig_pos_reg, upscaled_pos_reg, gray_reg)

full_qc.barrier(label='Upscaling Applied') # Add a barrier for clarity

# 5. Measure the registers
full_qc.measure(orig_pos_reg, c_orig_pos)
full_qc.measure(upscaled_pos_reg, c_up_pos)
full_qc.measure(gray_reg, c_gray)

# Draw final circuit
full_qc.draw(output='mpl', fold=-1, idle_wires=False)
plt.title("Quantum Nearest Neighbor Upscaling\n2x2 -> 4x4 (Combined Circuit)", fontsize=14, fontweight='bold')
plt.show()

# Optional: Run the circuit using Aer simulator
backend = Aer.get_backend('qasm_simulator')
job = backend.run(full_qc, shots=1024)
result = job.result()
counts = result.get_counts()
print("\nMeasurement Results:")
for k, v in counts.items():
    print(f"{k}: {v}")