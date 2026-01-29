#!/usr/bin/env python3
"""
Test showing the difference between blocky vs smooth preview visualization
"""
import sys
sys.path.insert(0, '/workspaces/QSVT_Hack')

import numpy as np
import matplotlib.pyplot as plt

print("Comparing Blocky vs Smooth Preview Visualization")
print("=" * 70)

# Simulate coarse grid (what was shown as blocky squares)
nx, ny = 3, 3
x_coarse = np.linspace(0, 1, nx, endpoint=False)
y_coarse = np.linspace(0, 1, ny, endpoint=False)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

# 2D Gaussian
center_x, center_y = 0.5, 0.5
width = 10.0
u0_coarse = np.exp(-width * ((X_coarse - center_x)**2 + (Y_coarse - center_y)**2))
u0_coarse_norm = u0_coarse / np.linalg.norm(u0_coarse.flatten())

print(f"\nCoarse Grid (Quantum Computation):")
print(f"  Grid points: {nx}×{ny} = {nx*ny}")
print(f"  Shape: {u0_coarse_norm.shape}")
print(f"  Values: {u0_coarse_norm.flatten()}")

# Fine grid for smooth preview
x_fine = np.linspace(0, 1, max(128, nx*32), endpoint=False)
y_fine = np.linspace(0, 1, max(128, ny*32), endpoint=False)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

u0_fine = np.exp(-width * ((X_fine - center_x)**2 + (Y_fine - center_y)**2))
u0_fine_norm = u0_fine / np.linalg.norm(u0_fine.flatten())

print(f"\nFine Grid (Smooth Preview):")
print(f"  Grid points: {len(x_fine)}×{len(y_fine)} = {len(x_fine)*len(y_fine)}")
print(f"  Shape: {u0_fine_norm.shape}")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='white')

# Blocky version
ax = axes[0]
u0_display_coarse = u0_coarse_norm / (np.max(u0_coarse_norm) + 1e-10)
im1 = ax.imshow(u0_display_coarse, cmap='viridis', origin='lower', 
                 extent=[0, 1, 0, 1], aspect='auto', interpolation='none')
ax.set_title("BEFORE: Blocky Squares\n(No interpolation)", fontsize=12, fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.colorbar(im1, ax=ax)

# Smooth version
ax = axes[1]
u0_display_fine = u0_fine_norm / (np.max(u0_fine_norm) + 1e-10)
im2 = ax.imshow(u0_display_fine, cmap='viridis', origin='lower', 
                 extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
ax.set_title("AFTER: Smooth Gaussian\n(Bilinear interpolation)", fontsize=12, fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.colorbar(im2, ax=ax)

# Difference visualization
ax = axes[2]
ax.text(0.5, 0.7, "✅ FIX APPLIED:", ha='center', va='center', fontsize=14, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.5, "Evaluate function on\nfine 128×128 grid\n(instead of coarse 3×3)",
        ha='center', va='center', fontsize=11, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.2, "Use bilinear interpolation\nfor smooth display",
        ha='center', va='center', fontsize=11, transform=ax.transAxes, family='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig('/tmp/preview_fix.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Comparison saved to /tmp/preview_fix.png")

print("\n" + "=" * 70)
print("FIX EXPLANATION:")
print("=" * 70)
print("""
PROBLEM:
  - Quantum computation uses coarse grid (3×3 or smaller)
  - Each point shown as large blocky square in imshow
  - All functions appeared blocky/ugly

SOLUTION:
  - Evaluate function on FINE grid (128×128) for preview only
  - Keep coarse grid for actual quantum circuit (still correct)
  - Use bilinear interpolation in imshow for smooth visualization
  - Now shows beautiful smooth Gaussian shapes while preserving quantum accuracy

KEY INSIGHT:
  - Preview is just for visualization - can use higher resolution
  - Quantum circuit uses actual coarse grid (still exact computation)
  - No loss of accuracy, just better visualization
""")
print("=" * 70)
