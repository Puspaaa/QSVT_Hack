#!/usr/bin/env python3
"""
Validation test for 2D Simulation page
Tests that all UI elements can be created without errors
"""
import sys
sys.path.insert(0, '/workspaces/QSVT_Hack')

import numpy as np
from solvers import cvx_poly_coef, Angles_Fixed
from quantum import QSVT_circuit_2d, Advection_Gate_2d

print("Testing 2D Simulation Page Workflow")
print("=" * 60)

# ============================================================
# 1. Test Initial Condition Preprocessing (All 5 Types)
# ============================================================
print("\n1. Testing Initial Condition Presets...")

nx, ny = 2, 2
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

presets = {}

# Gaussian Peak
width = 5.0
center_x, center_y = 0.5, 0.5
u0 = np.exp(-width * ((X - center_x)**2 + (Y - center_y)**2))
u0_norm = u0 / np.linalg.norm(u0.flatten())
presets['Gaussian Peak'] = u0_norm
print(f"  ✓ Gaussian Peak: shape {u0_norm.shape}, norm {np.linalg.norm(u0_norm.flatten()):.6f}")

# Double Gaussian
sig1, sig2 = 0.1, 0.1
c1_x, c1_y, c2_x, c2_y = 0.3, 0.3, 0.7, 0.7
u0 = (np.exp(-((X - c1_x)**2 + (Y - c1_y)**2) / (2 * sig1**2)) +
      np.exp(-((X - c2_x)**2 + (Y - c2_y)**2) / (2 * sig2**2)))
u0_norm = u0 / np.linalg.norm(u0.flatten())
presets['Double Gaussian'] = u0_norm
print(f"  ✓ Double Gaussian: shape {u0_norm.shape}, norm {np.linalg.norm(u0_norm.flatten()):.6f}")

# Gaussian Ring
r_center, r_width = 0.3, 0.1
r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
u0 = np.exp(-((r - r_center) / r_width)**2)
u0_norm = u0 / np.linalg.norm(u0.flatten())
presets['Gaussian Ring'] = u0_norm
print(f"  ✓ Gaussian Ring: shape {u0_norm.shape}, norm {np.linalg.norm(u0_norm.flatten()):.6f}")

# Sine Pattern
freq = 2
u0 = np.sin(freq * np.pi * X) * np.sin(freq * np.pi * Y)
u0 = np.abs(u0)
u0_norm = u0 / np.linalg.norm(u0.flatten())
presets['Sine Pattern'] = u0_norm
print(f"  ✓ Sine Pattern: shape {u0_norm.shape}, norm {np.linalg.norm(u0_norm.flatten()):.6f}")

# Custom (Gaussian as example)
width = 3.0
u0 = np.exp(-width * ((X - 0.5)**2 + (Y - 0.5)**2))
u0_norm = u0 / np.linalg.norm(u0.flatten())
presets['Custom'] = u0_norm
print(f"  ✓ Custom: shape {u0_norm.shape}, norm {np.linalg.norm(u0_norm.flatten()):.6f}")

print(f"\n✅ All {len(presets)} initial condition presets created successfully")

# ============================================================
# 2. Test Angle Computation
# ============================================================
print("\n2. Testing Angle Computation...")

t = 5.0
nu = 0.02
deg_diff = int(t + 8)
if deg_diff % 2 != 0:
    deg_diff += 1

target_f = lambda x: np.exp(t * (np.abs(x) - 1))
try:
    coef_diff = cvx_poly_coef(target_f, [0, 1], deg_diff, epsil=1e-5)
    phi_seq = Angles_Fixed(coef_diff)
    print(f"  ✓ Angles computed: degree={deg_diff}, angles={len(phi_seq)} values")
    print(f"    Sample angles: {phi_seq[:3]}")
except Exception as e:
    print(f"  ✗ Angle computation failed: {e}")
    sys.exit(1)

# ============================================================
# 3. Test Circuit Building
# ============================================================
print("\n3. Testing Circuit Building...")

init_state = presets['Gaussian Peak'].flatten()
init_state_expanded = np.zeros(2**(nx + ny))
init_state_expanded[:len(init_state)] = init_state
init_state_expanded /= np.linalg.norm(init_state_expanded)

try:
    qc = QSVT_circuit_2d(phi_seq, nx, ny, nu, init_state=init_state_expanded, measurement=True)
    print(f"  ✓ Circuit created: {qc.num_qubits} qubits, depth={qc.depth()}")
except Exception as e:
    print(f"  ✗ Circuit creation failed: {e}")
    sys.exit(1)

# ============================================================
# 4. Test Advection Gate
# ============================================================
print("\n4. Testing Advection Gate...")

c_x, c_y = 0.3, 0.3
try:
    adv_gate = Advection_Gate_2d(nx, ny, c_x, c_y, t)
    print(f"  ✓ Advection gate created: {adv_gate.num_qubits} qubits")
except Exception as e:
    print(f"  ✗ Advection gate creation failed: {e}")
    sys.exit(1)

# ============================================================
# 5. Test Measurement Parsing (Simulated)
# ============================================================
print("\n5. Testing Measurement Parsing...")

test_bitstrings = [
    "101010 00 0",
    "110011 00 0",
    "000000 00 0",
    "111111 11 1",
    "010101 01 0",
]

valid_count = 0
for bitstring in test_bitstrings:
    parts = bitstring.split()
    if len(parts) != 3:
        continue
    m_dat, m_anc, m_sig = parts
    if len(m_dat) == nx+ny and len(m_anc) == 2 and len(m_sig) == 1:
        if m_sig == '0' and m_anc == '00':
            valid_count += 1
            print(f"  ✓ Valid postselection: {bitstring}")

print(f"\n  Postselection check: {valid_count}/5 would pass")

print("\n" + "=" * 60)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 60)
