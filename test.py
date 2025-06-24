# # # # import numpy as np
# # # # from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# # # # from qiskit_aer import AerSimulator # Keep for local testing if needed, or remove
# # # # from qiskit.quantum_info import Statevector
# # # # from qiskit.circuit.library import RYGate
# # # # import matplotlib.pyplot as plt

# # # # # Import for IBM Quantum Runtime
# # # # from qiskit_ibm_runtime import QiskitRuntimeService
# # # # from qiskit_ibm_runtime.options import options # For general options if needed
# # # # from qiskit_ibm_runtime.options import SamplerOptions # Specific options for Sampler
# # # # from qiskit_ibm_runtime import Sampler # Import Sampler primitive

# # # # # --- Helper Functions (unchanged from your original code) ---

# # # # def hadamard(circ, n):
# # # #     """Applies Hadamard gate to a list of qubits."""
# # # #     for i in n:
# # # #         circ.h(i)

# # # # def change(state, new_state):
# # # #     """
# # # #     Compares two binary states and returns indices where they differ.
# # # #     Used to determine which qubits need to be flipped.
# # # #     """
# # # #     n = len(state)
# # # #     c = np.array([])
# # # #     for i in range(n):
# # # #         if state[i] != new_state[i]:
# # # #             c = np.append(c, int(i))
# # # #     return c.astype(int) if len(c) > 0 else c

# # # # def binary(circ, state, new_state, k):
# # # #     """
# # # #     Applies X gates to flip qubits based on differences between two binary states.
# # # #     (Note: This function is defined but not directly used in your main loop's FRQI logic,
# # # #     as `frqi_pixel` handles the position setting.)
# # # #     """
# # # #     c = change(state, new_state)
# # # #     if len(c) > 0:
# # # #         for i in c:
# # # #             circ.x(i)  # Correctly flip position qubits
# # # #     else:
# # # #         pass

# # # # def cnri(circ, control_qubits, target_qubit, theta):
# # # #     """
# # # #     Applies a controlled Ry gate (CRy) with multiple controls.
# # # #     This is a core component for encoding pixel intensity.
# # # #     """
# # # #     controls = len(control_qubits)
# # # #     cry = RYGate(theta).control(controls)  # Use theta for the rotation angle
# # # #     aux = np.append(control_qubits, target_qubit).tolist() # Combine control and target qubits
# # # #     circ.append(cry, aux)

# # # # def frqi_pixel(circ, control_qubits, target_qubit, angle, position_idx, num_qubits):
# # # #     """
# # # #     Encodes a single pixel's intensity and performs negation based on FRQI.

# # # #     Args:
# # # #         circ (QuantumCircuit): The quantum circuit to operate on.
# # # #         control_qubits (list): List of qubit indices for position encoding.
# # # #         target_qubit (int): Index of the intensity qubit.
# # # #         angle (float): The rotation angle (intensity) for the pixel.
# # # #         position_idx (int): The decimal index representing the pixel's position.
# # # #         num_qubits (int): The number of qubits used for position encoding.
# # # #     """
# # # #     # Set position qubits to |position_idx>
# # # #     # Convert decimal position_idx to a binary string of length num_qubits
# # # #     bin_idx = format(position_idx, f'0{num_qubits}b')
# # # #     for bit_val, qubit_idx in zip(bin_idx, control_qubits):
# # # #         if bit_val == '1':
# # # #             circ.x(qubit_idx) # Apply X gate if the corresponding bit is 1

# # # #     # Encode intensity with controlled Ry(theta)
# # # #     if angle > 0:
# # # #         cnri(circ, control_qubits, target_qubit, angle)
    
# # # #     circ.barrier(label="After_Encoding")
    
# # # #     # Negation: Reset intensity to |0> and apply controlled Ry(pi/2 - theta)
# # # #     negated_theta = np.pi/2 - angle
    
# # # #     # If the original angle was > 0, the intensity qubit was rotated.
# # # #     # To negate, we first "undo" the rotation to bring it back to |0> if it was |1>,
# # # #     # then apply the new rotation.
# # # #     if angle > 0:
# # # #         circ.x(target_qubit) # Flip to bring |1> to |0> for the negation rotation
    
# # # #     if negated_theta >= 0: # Ensure angle is non-negative
# # # #         cnri(circ, control_qubits, target_qubit, negated_theta)
    
# # # #     if angle > 0:
# # # #         circ.x(target_qubit) # Restore the basis if it was flipped for negation
    
# # # #     circ.barrier(label="After_Negation")

# # # # # --- Image Setup ---

# # # # # Input 2x2 binary image
# # # # image = np.array([
# # # #     [1, 0],
# # # #     [1, 0]
# # # # ])
# # # # print("\nOriginal Image (Binary):")
# # # # print(image)

# # # # n = image.shape[0]  # Assuming square image (n=2 for 2x2)
# # # # num_pixels = n * n  # Total number of pixels (4 for 2x2)
# # # # # Qubits for positions: ceil(log2(num_pixels))
# # # # # For 4 pixels, log2(4) = 2, so q = 2 control qubits
# # # # q = int(np.ceil(np.log2(num_pixels)))
# # # # print(f"Image size: {n}x{n}, using {q} control qubits")

# # # # # Compute angles for binary pixels (0 or 1)
# # # # # Binary 0 maps to angle 0, Binary 1 maps to angle pi/2
# # # # angles = np.pi/2 * image.flatten()

# # # # # Compute classical negated image for comparison
# # # # negated_image = 1 - image
# # # # print("\nNegated Image (Binary):")
# # # # print(negated_image)

# # # # # Process each pixel by defining its value, position index, and (row, col)
# # # # pixels = [
# # # #     (image[0,0], 0, (0,0)),  # Pixel at (0,0) -> position index 0 -> |00>
# # # #     (image[0,1], 1, (0,1)),  # Pixel at (0,1) -> position index 1 -> |01>
# # # #     (image[1,0], 2, (1,0)),  # Pixel at (1,0) -> position index 2 -> |10>
# # # #     (image[1,1], 3, (1,1))   # Pixel at (1,1) -> position index 3 -> |11>
# # # # ]

# # # # # --- IBM Quantum Runtime Setup ---

# # # # # 1. Load your IBM Quantum account.
# # # # # If you haven't saved your account yet, uncomment and run the line below once:
# # # # # from qiskit_ibm_runtime import QiskitRuntimeService
# # # # # QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_API_TOKEN', overwrite=True)
# # # # # Replace 'YOUR_API_TOKEN' with your actual token from quantum.ibm.com
# # # # # The 'ibm_quantum' channel is deprecated; 'ibm_cloud' is the current recommendation.
# # # # service = QiskitRuntimeService(
# # # #     token='88b7e01d0d72ecec68894a80350a71d8fe48257f19c7304a707a40e35c081a78f52e9b82de45041ea983b973cfd03de6dfe027c12f11a29e264b8f12644c5ab3' # ** IMPORTANT: Replace with your actual IBM Quantum API token **
# # # # )

# # # # # 2. Choose a backend.
# # # # # For real hardware, you would typically use:
# # # # # backend = service.least_busy(operational=True, simulator=False)
# # # # # Or specify a specific backend name:
# # # # # backend = service.get_backend('ibm_osaka')

# # # # # For cloud simulator (recommended for testing and statevector access):
# # # # backend = service.backend('ibm_brisbane')

# # # # # 3. Configure Sampler options.
# # # # # Use SamplerOptions for Sampler primitive specific configurations.
# # # # options = SamplerOptions()
# # # # # Removed options.resilience_level and options.optimization_level
# # # # # as they are not directly supported by SamplerOptions.
# # # # options.execution.shots = 999999 # Number of measurement shots for statistical results

# # # # # 4. Initialize the Sampler primitive.
# # # # sampler = Sampler(backend=backend, options=options)

# # # # # --- Pixel Processing Loop ---

# # # # for pixel_value, pos_idx, pos in pixels:
# # # #     # Quantum Circuit Setup for each pixel
# # # #     # q + 1 qubits: 'q' for position, +1 for intensity qubit
# # # #     qr = QuantumRegister(q + 1, 'q')
# # # #     cr = ClassicalRegister(1, 'c') # Classical register to measure the intensity qubit
# # # #     qc = QuantumCircuit(qr, cr)
    
# # # #     control_qubits = list(range(q)) # Qubits 0 to q-1 are control qubits for position
# # # #     target_qubit = q # The last qubit is the intensity qubit
    
# # # #     # Build FRQI circuit for this pixel
# # # #     angle = angles[pos_idx] # Get the intensity angle for the current pixel
# # # #     frqi_pixel(qc, control_qubits, target_qubit, angle, pos_idx, q)
    
# # # #     # Measure the intensity qubit to get the negated pixel value
# # # #     qc.measure(target_qubit, cr)
    
# # # #     # Print the quantum circuit diagram
# # # #     print(f"\nFRQI Circuit for Pixel {pixel_value} at Position {pos}:")
# # # #     print(qc.draw(output="text"))
    
# # # #     # Run the circuit using the Sampler primitive
# # # #     job = sampler.run(qc)
# # # #     result = job.result() # Wait for the job to complete and get results
    
# # # #     # --- Handling Statevector Information ---
# # # #     # Note: When running on a Sampler primitive (especially a cloud one),
# # # #     # you typically get measurement outcomes (counts/probabilities), not direct statevectors.
# # # #     # To get statevectors, you would run on a statevector simulator backend *without* measurements
# # # #     # and use `qc.save_statevector()`.
# # # #     print("\nStatevector BEFORE negation: (Not directly available from Runtime Primitives with measurements)")
# # # #     print("Statevector AFTER negation: (Not directly available from Runtime Primitives with measurements)")
# # # #     print("To get statevectors, run on 'simulator_statevector' backend *without* measurements and use qc.save_statevector()")

# # # #     # --- Estimate Negated Pixel Value from Measurement Counts ---
# # # #     # The Sampler returns quasi_dists, which are probability distributions.
# # # #     # For a single measured qubit, '0' or '1' are the possible outcomes.
# # # #     counts_prob = result.quasi_dists[0].binary_probabilities()
    
# # # #     # Convert probabilities back to "counts" for your original calculation logic
# # # #     total_shots = options.execution.shots
# # # #     count_1 = counts_prob.get('1', 0) * total_shots
# # # #     count_0 = counts_prob.get('0', 0) * total_shots
    
# # # #     total = count_0 + count_1
# # # #     negated_pixel = 0
# # # #     if total > 0:
# # # #         prob_1 = count_1 / total
# # # #         # Reconstruct the angle from the probability of |1>
# # # #         theta_prime = 2 * np.arcsin(np.sqrt(prob_1))
# # # #         # Convert the angle back to a binary pixel value (0 or 1)
# # # #         negated_pixel = int((theta_prime / (np.pi/2)) + 0.5) # Round to nearest 0 or 1
    
# # # #     print(f"\nExpected Negated Pixel Value: {1 - pixel_value}")
# # # #     print(f"Measured Negated Pixel Value: {negated_pixel}")

# # # # # --- Plotting Negated Image ---
# # # # # This part uses the classically negated image for plotting, as the quantum
# # # # # process is pixel-by-pixel and doesn't directly reconstruct an image.
# # # # plt.figure(figsize=(4, 2))
# # # # plt.imshow(negated_image, cmap='gray', vmin=0, vmax=1)
# # # # plt.title("Negated Image (Quantum)")
# # # # plt.axis('off')
# # # # plt.savefig('negated_image_binary.png')
# # # # plt.close()
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# # # pos = QuantumRegister(2, name='pos')         
# # # gray = QuantumRegister(4, name='gray')       
# # # c_pos = ClassicalRegister(2, name='c_pos')   
# # # c_gray = ClassicalRegister(4, name='c_gray') 
# # # qc = QuantumCircuit(pos, gray, c_pos, c_gray)

# # # qc.h(pos)

# # # def apply_grayscale(qc, pos_val, gray_val):
    
# # #     for i, bit in enumerate(f"{pos_val:02b}"):
# # #         if bit == '0':
# # #             qc.x(pos[i])

# # #     control_qubits = [pos[0], pos[1]]

# # #     for i, bit in enumerate(f"{gray_val:04b}"):  
# # #         if bit == '1':
# # #             qc.mcx(control_qubits, gray[3 - i])  

# # #     for i, bit in enumerate(f"{pos_val:02b}"):
# # #         if bit == '0':
# # #             qc.x(pos[i])

# # # apply_grayscale(qc, 0b00, 0)   # (0,0) = 0  -> 0000
# # # apply_grayscale(qc, 0b01, 5)   # (0,1) = 5  -> 0101
# # # apply_grayscale(qc, 0b10, 10)  # (1,0) = 10 -> 1010
# # # apply_grayscale(qc, 0b11, 15)  # (1,1) = 15 -> 1111

# # # qc.measure(pos, c_pos)
# # # qc.measure(gray, c_gray)

# # # print("\n--- Quantum Circuit for Input [[0, 5], [10, 15]] ---")
# # # fig = qc.draw('mpl', style={'backgroundcolor': '#ffffff'})
# # # fig.suptitle('Quantum Circuit for Input Matrix [[0, 5], [10, 15]]', fontsize=16)
# # # plt.figure(fig.number)
# # # plt.tight_layout()
# # # plt.show()
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# # # Define image matrix and upscale function
# # original = np.array([[0, 5], [10, 15]])

# # def nearest_neighbor_upscale(img):
# #     rows, cols = img.shape
# #     upscale = np.zeros((rows * 2, cols * 2), dtype=int)
# #     for i in range(rows):
# #         for j in range(cols):
# #             val = img[i, j]
# #             upscale[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = val
# #     return upscale

# # upscaled = nearest_neighbor_upscale(original)

# # # Function to add quantum image operations to a circuit
# # def add_quantum_image_operations(qc, matrix, pos_reg, gray_reg):
# #     """Add quantum image operations to an existing circuit"""
# #     num_pos_qubits = len(pos_reg)
    
# #     # Apply Hadamard gates to position qubits
# #     for qubit in pos_reg:
# #         qc.h(qubit)
    
# #     def apply_grayscale(pos_val, gray_val):
# #         bin_pos = f"{pos_val:0{num_pos_qubits}b}"
        
# #         # Set up position state
# #         for i, bit in enumerate(bin_pos):
# #             if bit == '0':
# #                 qc.x(pos_reg[i])
        
# #         # Apply controlled operations for grayscale value
# #         gray_bin = f"{gray_val:04b}"
# #         for i, bit in enumerate(gray_bin):
# #             if bit == '1':
# #                 if num_pos_qubits == 1:
# #                     qc.cx(pos_reg[0], gray_reg[i])
# #                 else:
# #                     qc.mcx(pos_reg[:], gray_reg[i])
        
# #         # Reset position qubits
# #         for i, bit in enumerate(bin_pos):
# #             if bit == '0':
# #                 qc.x(pos_reg[i])
    
# #     # Apply grayscale encoding for each pixel
# #     index = 0
# #     for i in range(matrix.shape[0]):
# #         for j in range(matrix.shape[1]):
# #             apply_grayscale(index, matrix[i][j])
# #             index += 1

# # # Calculate qubit requirements
# # original_pos_qubits = int(np.ceil(np.log2(original.size)))
# # upscaled_pos_qubits = int(np.ceil(np.log2(upscaled.size)))
# # max_pos_qubits = max(original_pos_qubits, upscaled_pos_qubits)
# # gray_qubits = 4

# # print(f"Original matrix:\n{original}")
# # print(f"Original matrix shape: {original.shape}")
# # print(f"Original image requires {original_pos_qubits} position qubits")

# # print(f"\nUpscaled matrix:\n{upscaled}")
# # print(f"Upscaled matrix shape: {upscaled.shape}")
# # print(f"Upscaled image requires {upscaled_pos_qubits} position qubits")

# # print(f"\nCombined circuit will use {max_pos_qubits} position qubits and {gray_qubits} grayscale qubits")

# # # Create registers for the combined circuit
# # pos_reg = QuantumRegister(max_pos_qubits, name='pos')
# # gray_reg = QuantumRegister(gray_qubits, name='gray')
# # c_pos_reg = ClassicalRegister(max_pos_qubits, name='c_pos')
# # c_gray_reg = ClassicalRegister(gray_qubits, name='c_gray')

# # # Create the combined circuit
# # combined_qc = QuantumCircuit(pos_reg, gray_reg, c_pos_reg, c_gray_reg)

# # # Section 1: Original Image Processing
# # print("\n--- Adding Original Image Processing ---")

# # # Add original image operations using only the required qubits
# # original_pos_subset = pos_reg[:original_pos_qubits]
# # add_quantum_image_operations(combined_qc, original, original_pos_subset, gray_reg)

# # # Measure original results
# # combined_qc.measure(original_pos_subset, c_pos_reg[:original_pos_qubits])
# # combined_qc.measure(gray_reg, c_gray_reg)

# # # Add barrier to separate the two sections visually
# # combined_qc.barrier()

# # # Reset qubits for the upscaled section
# # combined_qc.reset(pos_reg)
# # combined_qc.reset(gray_reg)

# # # Add another barrier after reset
# # combined_qc.barrier()

# # # Section 2: Upscaled Image Processing
# # print("--- Adding Upscaled Image Processing ---")

# # # Add upscaled image operations using only the required qubits
# # upscaled_pos_subset = pos_reg[:upscaled_pos_qubits]
# # add_quantum_image_operations(combined_qc, upscaled, upscaled_pos_subset, gray_reg)

# # # Measure upscaled results
# # combined_qc.measure(upscaled_pos_subset, c_pos_reg[:upscaled_pos_qubits])
# # combined_qc.measure(gray_reg, c_gray_reg)

# # print(f"\nCombined circuit created successfully!")
# # print(f"Total qubits: {combined_qc.num_qubits}")
# # print(f"Total classical bits: {combined_qc.num_clbits}")
# # print(f"Circuit depth: {combined_qc.depth()}")
# # print(f"Number of operations: {len(combined_qc.data)}")

# # # Create a custom drawing function for better visualization
# # def draw_combined_circuit(qc, title="Combined Quantum Circuit"):
# #     """Draw the circuit with custom styling for better clarity"""
# #     try:
# #         # Set up the plot with large figure size
# #         fig = plt.figure(figsize=(24, 14))
        
# #         # Custom style for better visibility
# #         style = {
# #             'backgroundcolor': '#ffffff',
# #             'textcolor': '#000000',
# #             'gatefacecolor': '#ffffff',
# #             'gateedgecolor': '#000000',
# #             'barrierfacecolor': '#ff6b6b',
# #             'cregbundle': True,
# #             'compress': False,
# #             'margin': [2, 0, 2, 0],
# #             'creglinestyle': 'solid',
# #             'displaytext': {
# #                 'H': 'H',
# #                 'X': 'X', 
# #                 'measure': 'M',
# #                 'reset': 'R'
# #             }
# #         }
        
# #         # Draw the circuit
# #         ax = qc.draw('mpl', style=style, fold=-1)
        
# #         # Set title
# #         plt.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
        
# #         # Add section labels with arrows
# #         fig_width = fig.get_figwidth()
        
# #         # Add text annotations for sections
# #         plt.figtext(0.25, 0.88, 'Original Image Processing\n(2×2 matrix → 4 pixels)', 
# #                    fontsize=14, fontweight='bold', ha='center',
# #                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
# #         plt.figtext(0.75, 0.88, 'Upscaled Image Processing\n(4×4 matrix → 16 pixels)', 
# #                    fontsize=14, fontweight='bold', ha='center',
# #                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
# #         # Add explanation text
# #         explanation = ("This quantum circuit processes both original and upscaled images sequentially.\n"
# #                       "Barriers separate the sections, and reset gates clear qubits between processes.")
# #         plt.figtext(0.5, 0.02, explanation, fontsize=12, ha='center', style='italic',
# #                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
# #         plt.tight_layout()
# #         plt.subplots_adjust(top=0.85, bottom=0.15)
# #         plt.show()
        
# #         return True
        
# #     except Exception as e:
# #         print(f"Error drawing circuit with matplotlib: {e}")
# #         return False

# # # Draw the combined circuit
# # print("\n--- Visualizing Combined Quantum Circuit ---")
# # success = draw_combined_circuit(combined_qc, "Quantum Image Processing: Original + Upscaled (Side by Side)")

# # if not success:
# #     print("Matplotlib visualization failed. Trying text representation...")
# #     try:
# #         print("\nCircuit text representation:")
# #         print(combined_qc.draw(output='text', fold=120))
# #     except Exception as e:
# #         print(f"Text representation also failed: {e}")
# #         print("\nBasic circuit information:")
# #         print(f"Registers: {[reg.name + '[' + str(reg.size) + ']' for reg in combined_qc.qregs]}")
# #         print(f"Classical registers: {[reg.name + '[' + str(reg.size) + ']' for reg in combined_qc.cregs]}")

# # # Create and display individual circuits for comparison
# # def create_individual_circuit(matrix, name):
# #     """Create a standalone circuit for a single matrix"""
# #     num_pos_qubits = int(np.ceil(np.log2(matrix.size)))
    
# #     pos = QuantumRegister(num_pos_qubits, name=f'pos')
# #     gray = QuantumRegister(4, name=f'gray')
# #     c_pos = ClassicalRegister(num_pos_qubits, name=f'c_pos')
# #     c_gray = ClassicalRegister(4, name=f'c_gray')
    
# #     qc = QuantumCircuit(pos, gray, c_pos, c_gray)
# #     add_quantum_image_operations(qc, matrix, pos, gray)
# #     qc.measure(pos, c_pos)
# #     qc.measure(gray, c_gray)
    
# #     return qc

# # # Create individual circuits for reference
# # print("\n--- Creating Individual Circuits for Reference ---")
# # qc_original_individual = create_individual_circuit(original, "Original")
# # qc_upscaled_individual = create_individual_circuit(upscaled, "Upscaled")

# # print(f"Original individual circuit: {qc_original_individual.num_qubits} qubits, depth {qc_original_individual.depth()}")
# # print(f"Upscaled individual circuit: {qc_upscaled_individual.num_qubits} qubits, depth {qc_upscaled_individual.depth()}")

# # # Draw individual circuits side by side for comparison
# # def draw_individual_circuits():
# #     try:
# #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))
        
# #         style_individual = {
# #             'backgroundcolor': '#ffffff',
# #             'textcolor': '#000000',
# #             'gatefacecolor': '#ffffff',
# #             'gateedgecolor': '#000000',
# #             'cregbundle': True,
# #             'margin': [1, 0, 1, 0]
# #         }
        
# #         # Draw original circuit
# #         qc_original_individual.draw('mpl', ax=ax1, style=style_individual)
# #         ax1.set_title('Original Matrix Circuit\n(2×2 matrix, 4 pixels, 2 pos qubits)', 
# #                      fontsize=14, fontweight='bold', pad=20)
        
# #         # Draw upscaled circuit
# #         qc_upscaled_individual.draw('mpl', ax=ax2, style=style_individual)
# #         ax2.set_title('Upscaled Matrix Circuit\n(4×4 matrix, 16 pixels, 4 pos qubits)', 
# #                      fontsize=14, fontweight='bold', pad=20)
        
# #         plt.suptitle('Individual Quantum Circuits Comparison', fontsize=16, fontweight='bold')
# #         plt.tight_layout()
# #         plt.subplots_adjust(top=0.9)
# #         plt.show()
        
# #         return True
        
# #     except Exception as e:
# #         print(f"Error drawing individual circuits: {e}")
# #         return False

# # print("\n--- Visualizing Individual Circuits ---")
# # individual_success = draw_individual_circuits()

# # if not individual_success:
# #     print("Individual circuit visualization failed.")
# #     try:
# #         print("\nOriginal circuit (text):")
# #         print(qc_original_individual.draw(output='text', fold=80))          
# #         print("\nUpscaled circuit (text):")
# #         print(qc_upscaled_individual.draw(output='text', fold=80))
# #     except:
# #         print("Text representation of individual circuits also failed.")

# # # Summary
# # print("\n" + "="*60)
# # print("QUANTUM IMAGE PROCESSING CIRCUIT SUMMARY")
# # print("="*60)
# # print(f"✓ Original matrix: {original.shape} → {original.size} pixels")
# # print(f"✓ Upscaled matrix: {upscaled.shape} → {upscaled.size} pixels")
# # print(f"✓ Combined circuit: {combined_qc.num_qubits} qubits, {combined_qc.num_clbits} classical bits")
# # print(f"✓ Circuit depth: {combined_qc.depth()}")
# # print(f"✓ Total operations: {len(combined_qc.data)}")
# # print("\nThe circuit successfully demonstrates quantum image processing")
# # print("for both original and upscaled images in a single, continuous flow.")
# # print("="*60)
# # import cv2, numpy as np
# # from PIL import Image
# # from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# # from qiskit_aer import AerSimulator
# # import matplotlib.pyplot as plt
# # import time

# # img = np.array([[ord(c)-48 if c.isdigit() else ord(c)-55 for c in line.strip()]
# #                 for line in open("lincon_rotated.txt")], dtype=np.uint8) * 7

# # cv2.imshow("Image from Text File", img)
# # cv2.waitKey(0)  
# # cv2.destroyAllWindows() 

# # cv2.imwrite("output_image.png", img)
# # --------------------------------------------------------------
# # import numpy as np
# # import cv2

# # try:
# #     # Read image data from text file
# #     with open("lincon_rotated.txt", 'r') as file:
# #         original_img = np.array([[ord(c)-48 if c.isdigit() else ord(c)-55 for c in line.strip()]
# #                                 for line in file], dtype=np.uint8) * 7
    
# #     print(f"Original image shape: {original_img.shape}")
    
# #     # Create upscaled version (4x scale using nearest neighbor interpolation)
# #     scale_factor = 400
# #     height, width = original_img.shape
# #     new_height, new_width = height * scale_factor, width * scale_factor
    
# #     upscaled_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
# #     print(f"Upscaled image shape: {upscaled_img.shape}")
    
# #     # Create side-by-side comparison
# #     # Resize upscaled image to match original height for side-by-side display
# #     display_width = int(new_width * height / new_height)
# #     upscaled_display = cv2.resize(upscaled_img, (display_width, height), interpolation=cv2.INTER_NEAREST)
    
# #     # Concatenate images horizontally
# #     side_by_side = np.hstack((original_img, upscaled_display))
    
# #     # Add labels to distinguish original vs upscaled
# #     labeled_img = side_by_side.copy()
    
# #     # Display the side-by-side comparison
# #     cv2.imshow("Original vs Upscaled", labeled_img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
    
# #     # Save the side-by-side comparison
# #     cv2.imwrite("original_vs_upscaled.tiff", labeled_img)
    
# #     # Also save individual images
# #     cv2.imwrite("original_image.png", original_img)
# #     cv2.imwrite("upscaled_image.png", upscaled_img)
    
# #     print("Images saved:")
# #     print("- original_vs_upscaled.png (side-by-side comparison)")
# #     print("- original_image.png (original)")
# #     print("- upscaled_image.png (full upscaled version)")
    
# # except FileNotFoundError:
# #     print("Error: 'lincon_rotated.txt' file not found!")
# # except Exception as e:
# #     print(f"Error: {e}")

# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # from PIL import Image

# # try:
# #     # Read image data from text file
# #     with open("lincon_rotated.txt", 'r') as file:
# #         original_img = np.array([[ord(c) - 48 if c.isdigit() else ord(c) - 55 for c in line.strip()]
# #                                  for line in file], dtype=np.uint8) * 7

# #     print(f"Original image shape: {original_img.shape}")

# #     # Create upscaled version (4x scale using nearest neighbor interpolation)
# #     scale_factor = 4
# #     height, width = original_img.shape
# #     new_height, new_width = height * scale_factor, width * scale_factor

# #     upscaled_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
# #     print(f"Upscaled image shape: {upscaled_img.shape}")

# #     # Convert NumPy arrays to PIL images for display via matplotlib
# #     orig_img = Image.fromarray(original_img)
# #     upscaled_display = Image.fromarray(upscaled_img)

# #     # Display using matplotlib
# #     plt.figure(figsize=(10, 5))
# #     plt.subplot(1, 2, 1)
# #     plt.imshow(orig_img, cmap='gray')
# #     plt.title("Original Image")
# #     plt.axis("off")

# #     plt.subplot(1, 2, 2)
# #     plt.imshow(upscaled_display, cmap='gray')
# #     plt.title("Upscaled Image")
# #     plt.axis("off")

# #     plt.tight_layout()
# #     plt.show()

# #     # Save .tiff with original image smaller than upscaled
# #     small_orig = cv2.resize(original_img, (width, height), interpolation=cv2.INTER_NEAREST)
# #     large_upscaled = cv2.resize(upscaled_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

# #     # Pad small image to match height of upscaled image for horizontal concat
# #     pad_top = (new_height - height) // 2
# #     pad_bottom = new_height - height - pad_top
# #     padded_orig = cv2.copyMakeBorder(small_orig, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

# #     # Concatenate padded original and upscaled
# #     side_by_side = np.hstack((padded_orig, large_upscaled))
# #     cv2.imwrite("original_vs_upscaled.tiff", side_by_side)

# #     # Save individual images
# #     cv2.imwrite("original_image.png", original_img)
# #     cv2.imwrite("upscaled_image.png", upscaled_img)

# #     print("Images saved:")
# #     print("- original_vs_upscaled.tiff (side-by-side with original smaller)")
# #     print("- original_image.png (original)")
# #     print("- upscaled_image.png (upscaled)")

# # except FileNotFoundError:
# #     print("Error: 'lincon_rotated.txt' file not found!")
# # except Exception as e:
# #     print(f"Error: {e}")

# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # from PIL import Image

# # try:
# #     # Read image data from text file
# #     with open("lincon_rotated.txt", 'r') as file:
# #         original_img = np.array([[ord(c) - 48 if c.isdigit() else ord(c) - 55 for c in line.strip()]
# #                                  for line in file], dtype=np.uint8) * 7

# #     print(f"Original image shape: {original_img.shape}")

# #     # Create upscaled version (4x scale using nearest neighbor interpolation)
# #     scale_factor = 4
# #     height, width = original_img.shape
# #     new_height, new_width = height * scale_factor, width * scale_factor

# #     upscaled_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
# #     print(f"Upscaled image shape: {upscaled_img.shape}")

# #     # Convert to PIL Images for display and saving
# #     orig_img = Image.fromarray(original_img)
# #     up_img = Image.fromarray(upscaled_img)

# #     # Display using matplotlib: orig smaller, upscaled larger
# #     plt.figure(figsize=(10, 5))
# #     plt.subplot(1, 2, 1)
# #     plt.imshow(orig_img.resize((128, 128)), cmap='gray')
# #     plt.title("Original Image")
# #     plt.axis("off")

# #     plt.subplot(1, 2, 2)
# #     plt.imshow(up_img.resize((256, 256)), cmap='gray')
# #     plt.title("Upscaled Image")
# #     plt.axis("off")

# #     plt.tight_layout()
# #     plt.show()

# #     # Prepare images for TIFF saving: maintain differing sizes
# #     small_orig = orig_img.resize((128, 128))
# #     large_up = up_img.resize((256, 256))

# #     # Create canvas
# #     canvas_w = small_orig.width + large_up.width
# #     canvas_h = max(small_orig.height, large_up.height)
# #     combined = Image.new('L', (canvas_w, canvas_h))

# #     # Paste side by side
# #     combined.paste(small_orig, (0, 0))
# #     combined.paste(large_up, (small_orig.width, 0))

# #     # Save side-by-side TIFF
# #     combined.save("original_vs_upscaled.tiff", format="TIFF")

# #     # Save side-by-side PNG as well (new addition)
# #     combined.save("original_vs_upscaled.png", format="PNG")

# #     # Save individual images
# #     orig_img.save("original_image.png")
# #     up_img.save("upscaled_image.png")

# #     print("Images saved:")
# #     print("- original_vs_upscaled.tiff (side-by-side with original smaller)")
# #     print("- original_vs_upscaled.png (side-by-side PNG version)")
# #     print("- original_image.png (original)")
# #     print("- upscaled_image.png (upscaled)")

# # except FileNotFoundError:
# #     print("Error: 'lincon_rotated.txt' file not found!")
# # except Exception as e:
# #     print(f"Error: {e}")
# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # from PIL import Image

# # try:
# #     # Read image data from text file
# #     with open("lincon_rotated.txt", 'r') as file:
# #         original_img = np.array([[ord(c) - 48 if c.isdigit() else ord(c) - 55 for c in line.strip()]
# #                                  for line in file], dtype=np.uint8) * 7

# #     print(f"Original image shape: {original_img.shape}")

# #     # Create upscaled version (4x scale using nearest neighbor interpolation)
# #     scale_factor = 4
# #     height, width = original_img.shape
# #     new_height, new_width = height * scale_factor, width * scale_factor

# #     upscaled_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
# #     print(f"Upscaled image shape: {upscaled_img.shape}")

# #     # Convert to PIL Images for display and saving
# #     orig_img = Image.fromarray(original_img)
# #     up_img = Image.fromarray(upscaled_img)

# #     # Display using matplotlib: orig smaller, upscaled larger
# #     plt.figure(figsize=(10, 5))
# #     plt.subplot(1, 2, 1)
# #     plt.imshow(orig_img.resize((128, 128)), cmap='gray')
# #     plt.title("Original Image")
# #     plt.axis("off")

# #     plt.subplot(1, 2, 2)
# #     plt.imshow(up_img.resize((256, 256)), cmap='gray')
# #     plt.title("Upscaled Image")
# #     plt.axis("off")

# #     plt.tight_layout()
# #     plt.show()

# #     # Prepare images for TIFF/PNG saving with white background
# #     small_orig = orig_img.resize((128, 128))
# #     large_up = up_img.resize((256, 256))

# #     # Create canvas with white background (255)
# #     canvas_w = small_orig.width + large_up.width
# #     canvas_h = max(small_orig.height, large_up.height)
# #     combined = Image.new('L', (canvas_w, canvas_h), color=255)

# #     # Paste side by side
# #     combined.paste(small_orig, (0, 0))
# #     combined.paste(large_up, (small_orig.width, 0))

# #     # Save side-by-side TIFF and PNG with white background
# #     combined.save("original_vs_upscaled.tiff", format="TIFF")
# #     combined.save("original_vs_upscaled.png", format="PNG")

# #     # Save individual images
# #     orig_img.save("original_image.png")
# #     up_img.save("upscaled_image.png")

# #     print("Images saved:")
# #     print("- original_vs_upscaled.tiff (side-by-side with white background)")
# #     print("- original_vs_upscaled.png (side-by-side PNG version)")
# #     print("- original_image.png (original)")
# #     print("- upscaled_image.png (upscaled)")

# # except FileNotFoundError:
# #     print("Error: 'lincon_rotated.txt' file not found!")
# # except Exception as e:
# #     print(f"Error: {e}")
# # ----------------------------------------------------------------------
# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# from qiskit_aer import AerSimulator
# from qiskit import transpile
# import numpy as np
# from numpy import pi
# import matplotlib.pyplot as plt

# # Define the 2x2 input matrix
# input_matrix = np.array([
#     [0, 5],
#     [10, 15]
# ])

# print("Input 2x2 Matrix:")
# print(input_matrix)

# # Convert grayscale values to quantum rotation angles
# # theta_yx = (grayscale / max_value) * (pi/2)
# max_value = 15  # Maximum value in our input matrix
# theta = {}
# for y in range(2):
#     for x in range(2):
#         theta[(y, x)] = (input_matrix[y, x] / max_value) * (pi / 2)

# print("\nRotation angles (theta):")
# for key, value in theta.items():
#     print(f"Position {key}: {value:.4f} radians")

# # Create quantum circuit
# qreg_x_prime = QuantumRegister(2, 'x_prime')
# qreg_y_prime = QuantumRegister(2, 'y_prime')
# qreg_anc = QuantumRegister(2, 'anc')
# qreg_color = QuantumRegister(1, 'color')
# creg_c = ClassicalRegister(1, 'c')
# circuit = QuantumCircuit(qreg_x_prime, qreg_y_prime, qreg_anc, qreg_color, creg_c)

# # Step 1: Create superposition for all positions
# circuit.h(qreg_x_prime[0])  # Create superposition for x'_0
# circuit.h(qreg_x_prime[1])  # Create superposition for x'_1
# circuit.h(qreg_y_prime[0])  # Create superposition for y'_0  
# circuit.h(qreg_y_prime[1])  # Create superposition for y'_1

# # Step 2: Position mapping using CNOT gates
# circuit.cx(qreg_x_prime[0], qreg_anc[0])  # x = x'_0 (map x coordinate)
# circuit.cx(qreg_y_prime[0], qreg_anc[1])  # y = y'_0 (map y coordinate)

# # Step 3: Color assignment for each pixel position
# # We need to apply controlled rotations for each position (y,x)

# # Position (0,0): value = 0, theta = 0 (no rotation needed)
# # Skip this as theta = 0

# # Position (0,1): value = 5, theta = 5/15 * pi/2 = pi/6
# theta_01 = theta[(0, 1)]
# if theta_01 > 0:
#     # Set ancilla to represent position (0,1): anc[0]=1, anc[1]=0
#     circuit.x(qreg_anc[0])  # Flip anc[0] to |1> for x=1
#     # Apply controlled rotations when both ancilla qubits match the position
#     circuit.mcry(2 * theta_01, [qreg_anc[1], qreg_anc[0]], qreg_color[0], None)
#     circuit.x(qreg_anc[0])  # Restore anc[0]

# # Position (1,0): value = 10, theta = 10/15 * pi/2 = pi/3
# theta_10 = theta[(1, 0)]
# if theta_10 > 0:
#     # Set ancilla to represent position (1,0): anc[0]=0, anc[1]=1
#     circuit.x(qreg_anc[1])  # Flip anc[1] to |1> for y=1
#     # For position (1,0), we need anc[0]=0 and anc[1]=1
#     circuit.x(qreg_anc[0])  # Temporarily flip anc[0] to make it |0> when controlled
#     circuit.mcry(2 * theta_10, [qreg_anc[0], qreg_anc[1]], qreg_color[0], None)
#     circuit.x(qreg_anc[0])  # Restore anc[0]
#     circuit.x(qreg_anc[1])  # Restore anc[1]

# # Position (1,1): value = 15, theta = 15/15 * pi/2 = pi/2
# theta_11 = theta[(1, 1)]
# if theta_11 > 0:
#     # Set ancilla to represent position (1,1): anc[0]=1, anc[1]=1
#     circuit.x(qreg_anc[0])  # Flip anc[0] to |1> for x=1
#     circuit.x(qreg_anc[1])  # Flip anc[1] to |1> for y=1
#     circuit.mcry(2 * theta_11, [qreg_anc[1], qreg_anc[0]], qreg_color[0], None)
#     circuit.x(qreg_anc[0])  # Restore anc[0]
#     circuit.x(qreg_anc[1])  # Restore anc[1]

# # Alternative implementation using individual controlled rotations
# # This approach applies rotations based on ancilla states without requiring CCRY

# # Reset circuit for alternative approach
# circuit = QuantumCircuit(qreg_x_prime, qreg_y_prime, qreg_anc, qreg_color, creg_c)

# # Step 1: Create superposition
# circuit.h(qreg_x_prime[0])
# circuit.h(qreg_x_prime[1])
# circuit.h(qreg_y_prime[0])
# circuit.h(qreg_y_prime[1])

# # Step 2: Position mapping
# circuit.cx(qreg_x_prime[0], qreg_anc[0])
# circuit.cx(qreg_y_prime[0], qreg_anc[1])

# # Step 3: Apply rotations for each position using conditional logic
# # Position (0,1): anc[0]=1, anc[1]=0
# circuit.x(qreg_anc[0])  # Flip to |1>
# circuit.cry(2 * theta[(0, 1)], qreg_anc[1], qreg_color[0])  # Control on anc[1]=0
# circuit.cry(2 * theta[(0, 1)], qreg_anc[0], qreg_color[0])  # Control on anc[0]=1
# circuit.x(qreg_anc[0])  # Restore

# # Position (1,0): anc[0]=0, anc[1]=1
# circuit.x(qreg_anc[1])  # Flip to |1>
# circuit.cry(2 * theta[(1, 0)], qreg_anc[1], qreg_color[0])  # Control on anc[1]=1
# circuit.cry(2 * theta[(1, 0)], qreg_anc[0], qreg_color[0])  # Control on anc[0]=0
# circuit.x(qreg_anc[1])  # Restore

# # Position (1,1): anc[0]=1, anc[1]=1
# circuit.x(qreg_anc[0])  # Flip to |1>
# circuit.x(qreg_anc[1])  # Flip to |1>
# circuit.cry(2 * theta[(1, 1)], qreg_anc[1], qreg_color[0])  # Control on anc[1]=1
# circuit.cry(2 * theta[(1, 1)], qreg_anc[0], qreg_color[0])  # Control on anc[0]=1
# circuit.x(qreg_anc[0])  # Restore
# circuit.x(qreg_anc[1])  # Restore

# # Step 4: Measure the color qubit
# circuit.measure(qreg_color[0], creg_c[0])

# print(f"\nQuantum Circuit:")
# print(circuit.draw(output='text'))

# # Simulate the circuit
# simulator = AerSimulator()
# compiled_circuit = transpile(circuit, simulator)
# job = simulator.run(compiled_circuit, shots=8192)
# result = job.result()
# counts = result.get_counts()

# print(f"\nMeasurement counts: {counts}")

# # Reconstruct the 4x4 output matrix
# output_matrix = np.zeros((4, 4))

# # For nearest neighbor upscaling from 2x2 to 4x4
# for y_out in range(4):
#     for x_out in range(4):
#         # Map 4x4 coordinates back to 2x2 coordinates
#         y_in = y_out // 2
#         x_in = x_out // 2
#         output_matrix[y_out, x_out] = input_matrix[y_in, x_in]

# print(f"\nExpected 4x4 Output Matrix (Nearest Neighbor Upscaling):")
# print(output_matrix.astype(int))

# # Visualize both matrices
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# # Input matrix
# im1 = ax1.imshow(input_matrix, cmap='gray', vmin=0, vmax=15)
# ax1.set_title('Input 2x2 Matrix')
# ax1.set_xticks([0, 1])
# ax1.set_yticks([0, 1])
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# # Add value labels
# for i in range(2):
#     for j in range(2):
#         ax1.text(j, i, str(input_matrix[i, j]), ha='center', va='center', color='red', fontweight='bold')

# im2 = ax2.imshow(output_matrix, cmap='gray', vmin=0, vmax=15)
# ax2.set_title('Upscaled 4x4 Matrix')
# ax2.set_xticks([0, 1, 2, 3])
# ax2.set_yticks([0, 1, 2, 3])
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# # Add value labels
# for i in range(4):
#     for j in range(4):
#         ax2.text(j, i, str(int(output_matrix[i, j])), ha='center', va='center', color='red', fontweight='bold')

# plt.tight_layout()
# plt.colorbar(im1, ax=ax1)
# plt.colorbar(im2, ax=ax2)
# plt.show()

# print(f"\nRotation angles used:")
# print(f"theta(0,0) = {theta[(0,0)]:.4f} rad (value: 0)")
# print(f"theta(0,1) = {theta[(0,1)]:.4f} rad (value: 5)")
# print(f"theta(1,0) = {theta[(1,0)]:.4f} rad (value: 10)")
# print(f"theta(1,1) = {theta[(1,1)]:.4f} rad (value: 15)")
