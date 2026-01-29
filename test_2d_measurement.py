#!/usr/bin/env python3
"""Quick test to verify 2D measurement parsing"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Test parameters
nx, ny = 3, 3

# Create simple test circuit
sig = QuantumRegister(1, 'sig')
anc = QuantumRegister(2, 'anc')
x_reg = QuantumRegister(nx, 'x')
y_reg = QuantumRegister(ny, 'y')

c_sig = ClassicalRegister(1, 'm_sig')
c_anc = ClassicalRegister(2, 'm_anc')
c_dat = ClassicalRegister(nx + ny, 'm_dat')

qc = QuantumCircuit(sig, anc, x_reg, y_reg, c_sig, c_anc, c_dat)

# Initialize x and y to |010⟩ (flat_idx = 0 + 1*3 = 3)
qc.x(x_reg[1])  # Set x[1] = 1
qc.x(y_reg[0])  # Set y[0] = 1

# Measurements
qc.measure(sig, c_sig)
qc.measure(anc, c_anc)
qc.measure(x_reg, c_dat[:nx])
qc.measure(y_reg, c_dat[nx:])

# Run
backend = AerSimulator()
tqc = transpile(qc, backend, optimization_level=0)
counts = backend.run(tqc, shots=100).result().get_counts()

print("Test Circuit Measurement Results:")
print("=" * 60)
print(f"Grid size: {nx}×{ny}")
print(f"Total data bits: {nx + ny}")
print()

for bitstring, count in list(counts.items())[:5]:
    print(f"Bitstring: '{bitstring}'")
    print(f"  Count: {count}")
    parts = bitstring.split()
    print(f"  Parts: {parts}")
    print(f"  Num parts: {len(parts)}")
    if len(parts) == 3:
        m_dat, m_anc, m_sig = parts
        print(f"    m_dat ({len(m_dat)} bits): {m_dat}")
        print(f"    m_anc ({len(m_anc)} bits): {m_anc}")
        print(f"    m_sig ({len(m_sig)} bit): {m_sig}")
    print()

print("=" * 60)
print("✅ Measurement parsing test complete!")
