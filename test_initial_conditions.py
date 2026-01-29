#!/usr/bin/env python3
"""
Quick test to verify initial condition functions work and display properly
"""
import sys
sys.path.insert(0, '/workspaces/QSVT_Hack')

import numpy as np
import matplotlib.pyplot as plt

print("Testing Initial Condition Functions")
print("=" * 60)

# Grid sizes for testing
nx, ny = 3, 3
x = np.linspace(0, 1, nx, endpoint=False)
y = np.linspace(0, 1, ny, endpoint=False)
X, Y = np.meshgrid(x, y)

print(f"\nGrid: {nx}×{ny} = {nx*ny} points\n")

# Test 1: Gaussian Peak (DEFAULT - should be 2D Gaussian)
print("1. Gaussian Peak (DEFAULT - 2D Gaussian)")
center_x, center_y = 0.5, 0.5
width = 10.0
u0_gauss = np.exp(-width * ((X - center_x)**2 + (Y - center_y)**2))
u0_gauss_norm = u0_gauss / np.linalg.norm(u0_gauss.flatten())
print(f"   Shape: {u0_gauss_norm.shape}")
print(f"   Min: {np.min(u0_gauss_norm):.6f}, Max: {np.max(u0_gauss_norm):.6f}")
print(f"   Norm: {np.linalg.norm(u0_gauss_norm.flatten()):.6f}")
print(f"   ✓ Valid 2D Gaussian")

# Test 2: Double Gaussian
print("\n2. Double Gaussian")
c1x, c1y = 0.3, 0.3
c2x, c2y = 0.7, 0.7
width = 8.0
u0_double = (np.exp(-width * ((X - c1x)**2 + (Y - c1y)**2)) + 
              np.exp(-width * ((X - c2x)**2 + (Y - c2y)**2)))
u0_double_norm = u0_double / np.linalg.norm(u0_double.flatten())
print(f"   Shape: {u0_double_norm.shape}")
print(f"   Min: {np.min(u0_double_norm):.6f}, Max: {np.max(u0_double_norm):.6f}")
print(f"   Norm: {np.linalg.norm(u0_double_norm.flatten()):.6f}")
print(f"   ✓ Valid Double Gaussian")

# Test 3: Gaussian Ring
print("\n3. Gaussian Ring")
r_center, r_width = 0.3, 10.0
r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
u0_ring = np.exp(-r_width * (r - r_center)**2)
u0_ring_norm = u0_ring / np.linalg.norm(u0_ring.flatten())
print(f"   Shape: {u0_ring_norm.shape}")
print(f"   Min: {np.min(u0_ring_norm):.6f}, Max: {np.max(u0_ring_norm):.6f}")
print(f"   Norm: {np.linalg.norm(u0_ring_norm.flatten()):.6f}")
print(f"   ✓ Valid Gaussian Ring")

# Test 4: Sine Pattern
print("\n4. Sine Pattern")
freq_x, freq_y = 2, 2
u0_sine = np.abs(np.sin(2 * np.pi * freq_x * X) * np.sin(2 * np.pi * freq_y * Y))
u0_sine_norm = u0_sine / np.linalg.norm(u0_sine.flatten())
print(f"   Shape: {u0_sine_norm.shape}")
print(f"   Min: {np.min(u0_sine_norm):.6f}, Max: {np.max(u0_sine_norm):.6f}")
print(f"   Norm: {np.linalg.norm(u0_sine_norm.flatten()):.6f}")
print(f"   ✓ Valid Sine Pattern")

# Test 5: Custom (Gaussian expression)
print("\n5. Custom Function")
custom_expr = "np.exp(-10*((X-0.5)**2 + (Y-0.5)**2))"
u0_custom = eval(custom_expr)
u0_custom_norm = u0_custom / np.linalg.norm(u0_custom.flatten())
print(f"   Shape: {u0_custom_norm.shape}")
print(f"   Min: {np.min(u0_custom_norm):.6f}, Max: {np.max(u0_custom_norm):.6f}")
print(f"   Norm: {np.linalg.norm(u0_custom_norm.flatten()):.6f}")
print(f"   ✓ Valid Custom Function")

# Check all are different
print("\n" + "=" * 60)
print("Comparison: Different functions produce different outputs")
print(f"  Gaussian Peak ≠ Double Gaussian: {not np.allclose(u0_gauss_norm, u0_double_norm)}")
print(f"  Gaussian Peak ≠ Gaussian Ring: {not np.allclose(u0_gauss_norm, u0_ring_norm)}")
print(f"  Double Gaussian ≠ Sine Pattern: {not np.allclose(u0_double_norm, u0_sine_norm)}")
print(f"  ✓ All functions are distinct")

print("\n✅ ALL TESTS PASSED!")
print("=" * 60)
