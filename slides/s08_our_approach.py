"""Slide 8: Our Implementation & Architecture."""

import streamlit as st
from slides.components import slide_header, key_concept
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TITLE = "Our Implementation"


def render():
    slide_header("Our Implementation",
                 "Architecture, design decisions, and toolchain")

    # ── Architecture diagram ──
    st.markdown("### System Architecture")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Layer definitions: (label, color, y_center, boxes)
    layers = [
        ("Frontend", "#27ae60", 4.0, [
            ("Streamlit UI", 0.5), ("Interactive Plots", 4.5), ("Live Demos", 8.5),
        ]),
        ("Algorithm", "#4a90d9", 2.5, [
            ("solvers.py\nChebyshev + Angles", 0.5),
            ("quantum.py\nCircuit Builder", 4.5),
            ("simulation.py\nPDE Runner", 8.5),
        ]),
        ("Execution", "#7b61ff", 1.0, [
            ("Qiskit 1.x\nCircuit Compilation", 0.5),
            ("AerSimulator\nStatevector Backend", 4.5),
            ("measurements.py\nPostselection", 8.5),
        ]),
        ("Classical", "#e67e22", -0.5, [
            ("CVXPY\nPolynomial Optimization", 0.5),
            ("pyqsp\nAngle Finding", 4.5),
            ("NumPy/SciPy\nReference Solutions", 8.5),
        ]),
    ]

    for layer_label, color, y, boxes in layers:
        # Layer background
        rect = plt.Rectangle((-0.3, y - 0.5), 13.6, 1.0,
                               facecolor=color, alpha=0.08, edgecolor=color,
                               linewidth=1.5, linestyle='--')
        ax.add_patch(rect)
        ax.text(-0.5, y, layer_label, fontsize=9, fontweight='bold', color=color,
                ha='right', va='center', rotation=90)

        for label, x in boxes:
            box = mpatches.FancyBboxPatch((x, y - 0.35), 3.5, 0.7,
                                           boxstyle="round,pad=0.1", facecolor=color,
                                           alpha=0.75, edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(x + 1.75, y, label, fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white')

    # Arrows between layers
    for x_center in [2.25, 6.25, 10.25]:
        for y_start in [3.5, 2.0, 0.5]:
            ax.annotate("", xy=(x_center, y_start - 0.05),
                        xytext=(x_center, y_start + 0.45),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color='#555'))

    ax.set_xlim(-2, 14)
    ax.set_ylim(-1.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("QSVT Scientific Computing — Architecture", fontsize=14, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── Key design decisions ──
    st.markdown("### Key Design Decisions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(r"""
**Block Encoding: LCU with 2 ancilla qubits**
- Diffusion operator decomposes into $a_0 I + a_+ S + a_- S^\dagger$
- Only 3 unitaries → minimal ancilla overhead
- QFT-based cyclic shift operator is $O(n^2)$ gates

**QSVT Angle Finding: `pyqsp` library**
- Uses symmetric QSP angle-finding algorithm
- Chebyshev coefficients → QSVT rotation angles
- Classical preprocessing: $O(d^3)$

**Polynomial Optimization: CVXPY**
- Constrained optimization: $|P(x)| \leq 1 - \epsilon$ 
- Enforces parity (even/odd) automatically
- Uses SCS solver for semidefinite programs
""")

    with col2:
        st.markdown(r"""
**Simulation: Qiskit + AerSimulator**
- Full circuit simulation (not tensor-network shortcuts)
- Statevector backend for exact amplitudes
- Shot-based sampling for realistic noise modeling

**Time Integration: Split-Step**
- Lie-Trotter splitting: diffusion (QSVT) + advection (QFT)
- CFL condition: $\Delta t = 0.9 \cdot \Delta x^2 / (2\nu)$
- Multiple time steps with circuit reuse

**Validation: 3-Way Comparison**
- Exact (Fourier/spectral): analytical reference
- Classical (finite difference): standard numerical method
- Quantum (QSVT): our implementation
""")

    st.markdown("---")

    # ── Complexity table ──
    st.markdown("### Resource Scaling")
    st.markdown(r"""
| Component | Classical | Quantum (QSVT) |
|-----------|-----------|----------------|
| **State storage** | $O(N)$ memory | $n = \log_2 N$ qubits |
| **Matrix-vector multiply** | $O(N \cdot s)$ | $O(1)$ query to $U_A$ |
| **Polynomial $P(A)$ of degree $d$** | $O(d \cdot N \cdot s)$ | $O(d)$ queries |
| **Total per time step** | $O(d \cdot N)$ | $O(d \cdot \text{poly}(n))$ |
| **Readout** | $O(1)$ per entry | $O(N/\epsilon^2)$ for full tomography |
""")

    st.warning(
        "**Caveat:** Full state readout negates the quantum advantage. "
        "The speedup applies when extracting *observables* (e.g., integrals, expectation values) "
        "rather than the entire solution vector."
    )

    key_concept(
        "Our implementation is a <b>complete, working pipeline</b>: "
        "from PDE specification → polynomial approximation → QSVT angle finding → "
        "circuit construction → simulation → postselection → visualization. "
        "All running in your browser via Streamlit."
    )
