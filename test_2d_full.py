#!/usr/bin/env python3
"""Quick test of 2D QSVT circuit execution"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from quantum import QSVT_circuit_2d, Advection_Gate_2d
from solvers import Angles_Fixed, cvx_poly_coef

# Test parameters
nx, ny = 2, 2
nu = 0.02
c_x, c_y = 0.3, 0.3
t = 5
shots = 1000

print("Testing 2D QSVT Circuit")
print("=" * 60)
print(f"Grid: {nx}×{ny}, Viscosity: {nu}")
print(f"Advection: c_x={c_x}, c_y={c_y}, t={t}")
print()

# Compute angles
print("Computing QSVT angles...")
deg_diff = int(t + 8)
if deg_diff % 2 != 0:
    deg_diff += 1
print(f"  Degree: {deg_diff}")

target_f = lambda x: np.exp(t * (np.abs(x) - 1))
coeffs_diff = cvx_poly_coef(target_f, [0, 1], deg_diff, epsil=1e-5)
print(f"  Coefficients computed: {len(coeffs_diff)} values")

phi_seq = Angles_Fixed(coeffs_diff)
print(f"  Angles computed: {len(phi_seq)} values")
print()

# Initial state - for 2D grid we need full 2^(nx+ny) state
print("Preparing initial state...")
# Create a proper superposition over all basis states
# For 2×2 grid: 4 data qubits = 16 basis states
# But we only have nx*ny = 4 grid points
# So we need to embed the 4D state into a 16D space... which doesn't make sense

# Actually, for QSVT with 2D grid:
# We should have nx + ny = 4 qubits total for data
# But these encode nx × ny = 4 grid points via the flattening idx = x + y*nx
# So a uniform state across all grid points is |0⟩⊗|0⟩ + |1⟩⊗|0⟩ + |0⟩⊗|1⟩ + |1⟩⊗|1⟩
# This is a 4D superposition (2 qubits x 2 qubits), not a full 16D one

# Let's create the proper initial superposition
n_qubits = nx + ny
n_states = 2**n_qubits
init_state = np.ones(n_states) / np.sqrt(n_states)  # Uniform superposition

print(f"  State dimension: {len(init_state)}")
print(f"  Expected qubits: {n_qubits}")
print(f"  State norm: {np.linalg.norm(init_state):.6f}")
print()

# Build circuit
print("Building 2D QSVT circuit...")
qc = QSVT_circuit_2d(phi_seq, nx, ny, nu, init_state=init_state, measurement=True)
print(f"  Circuit qubits: {qc.num_qubits}")
print(f"  Circuit depth: {qc.depth()}")
print()

# Add advection
print("Adding advection gate...")
physical_time = t * (0.9 * min(1/nx**2, 1/ny**2) / (4 * nu))
adv_gate = Advection_Gate_2d(nx, ny, c_x, c_y, physical_time)
x_reg = None
y_reg = None
for qreg in qc.qregs:
    if qreg.name == 'x':
        x_reg = qreg
    elif qreg.name == 'y':
        y_reg = qreg
if x_reg and y_reg:
    qc.append(adv_gate, list(x_reg) + list(y_reg))
print(f"  Circuit depth after advection: {qc.depth()}")
print()

# Run
print("Running circuit on simulator...")
backend = AerSimulator()
tqc = transpile(qc, backend, optimization_level=0)
job = backend.run(tqc, shots=shots)
counts = job.result().get_counts()
print(f"  Shot count: {len(counts)} unique outcomes")
print()

# Parse results
print("Parsing measurement results...")
prob_dist = np.zeros(nx * ny)
total_valid = 0
parse_errors = 0

for bitstring, count in counts.items():
    try:
        parts = bitstring.split()
        
        if len(parts) != 3:
            parse_errors += 1
            continue
        
        m_dat = parts[0]
        m_anc = parts[1]
        m_sig = parts[2]
        
        if len(m_dat) != nx + ny or len(m_anc) != 2 or len(m_sig) != 1:
            parse_errors += 1
            continue
        
        if m_sig == '0' and m_anc == '00':
            flat_idx = int(m_dat, 2)
            if 0 <= flat_idx < nx * ny:
                prob_dist[flat_idx] += count
                total_valid += count
    except:
        parse_errors += 1

print(f"  Total valid (postselected): {total_valid}/{shots}")
print(f"  Success rate: {100*total_valid/shots:.1f}%")
print(f"  Parse errors: {parse_errors}")
print()

if total_valid > 0:
    prob_dist_norm = prob_dist / total_valid
    print("✅ 2D QSVT circuit execution successful!")
    print(f"  Output probabilities:\n{prob_dist_norm}")
else:
    print("❌ No valid outcomes after postselection")

print()
print("=" * 60)
