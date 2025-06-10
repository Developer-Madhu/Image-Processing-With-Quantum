import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

pos = QuantumRegister(2, name='pos')         
gray = QuantumRegister(4, name='gray')       
c_pos = ClassicalRegister(2, name='c_pos')   
c_gray = ClassicalRegister(4, name='c_gray') 
qc = QuantumCircuit(pos, gray, c_pos, c_gray)

qc.h(pos)

def apply_grayscale(qc, pos_val, gray_val):
    
    for i, bit in enumerate(f"{pos_val:02b}"):
        if bit == '0':
            qc.x(pos[i])

    control_qubits = [pos[0], pos[1]]

    for i, bit in enumerate(f"{gray_val:04b}"):  
        if bit == '1':
            qc.mcx(control_qubits, gray[3 - i])  

    for i, bit in enumerate(f"{pos_val:02b}"):
        if bit == '0':
            qc.x(pos[i])

apply_grayscale(qc, 0b00, 0)   # (0,0) = 0  -> 0000
apply_grayscale(qc, 0b01, 5)   # (0,1) = 5  -> 0101
apply_grayscale(qc, 0b10, 10)  # (1,0) = 10 -> 1010
apply_grayscale(qc, 0b11, 15)  # (1,1) = 15 -> 1111

qc.measure(pos, c_pos)
qc.measure(gray, c_gray)

print("\n--- Quantum Circuit for Input [[0, 5], [10, 15]] ---")
fig = qc.draw('mpl', style={'backgroundcolor': '#ffffff'})
fig.suptitle('Quantum Circuit for Input Matrix [[0, 5], [10, 15]]', fontsize=16)
plt.figure(fig.number)
plt.tight_layout()
plt.show()
