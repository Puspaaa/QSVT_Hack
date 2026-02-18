"""Slide 6: QSVT for PDEs — Operator Splitting."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import slide_header, reference, key_concept, reference_list

TITLE = "QSVT for PDEs"


def render():
    slide_header("Solving PDEs with QSVT",
                 "Advection-diffusion via operator splitting")

    st.markdown(r"""
### The Advection-Diffusion Equation

$$\frac{\partial u}{\partial t} = \nu \nabla^2 u \;-\; \mathbf{c} \cdot \nabla u$$

- **Diffusion** ($\nu \nabla^2 u$): smoothing / heat conduction  
- **Advection** ($-\mathbf{c} \cdot \nabla u$): transport by a velocity field  
- Periodic boundary conditions on $[0, 1)^d$
""")

    st.markdown("---")

    # ── Operator Splitting ──
    st.markdown("### Operator Splitting Strategy")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(r"""
        #### Step 1: Split
        $$e^{t(\nu\nabla^2 - \mathbf{c}\cdot\nabla)} \approx e^{t\nu\nabla^2} \cdot e^{-t\mathbf{c}\cdot\nabla}$$
        
        First-order Lie-Trotter splitting.  
        Each part is handled by a *different* quantum technique.
        """)

    with col2:
        st.markdown(r"""
        #### Step 2: Diffusion (QSVT)
        $$e^{t\nu\nabla^2} \approx P(A_{\text{diff}})$$
        
        - Block-encode the discrete Laplacian via **LCU**
        - Apply QSVT with even polynomial $P(x) = e^{t(|x|-1)}$
        - Uses 2 ancilla + 1 signal qubit
        """)

    with col3:
        st.markdown(r"""
        #### Step 3: Advection (QFT)
        $$e^{-tc\partial_x} \to \text{phase kicks in Fourier space}$$
        
        - Apply QFT to data register
        - Diagonal phase rotations: $e^{-2\pi i k c t}$
        - Apply inverse QFT
        - **No polynomial approximation needed!**
        """)

    st.markdown("---")

    # ── Pipeline diagram ──
    st.markdown("### Full Pipeline")

    fig, ax = plt.subplots(figsize=(14, 3))

    boxes = [
        ("Encode\n$|u_0\\rangle$", "#27ae60", 0),
        ("Block\nEncoding\n(LCU)", "#4a90d9", 1),
        ("QSVT\nDiffusion\n$P(A)$", "#7b61ff", 2),
        ("QFT\nAdvection", "#e67e22", 3),
        ("Measure\n& Postselect", "#e74c3c", 4),
        ("Repeat\n(time steps)", "#95a5a6", 5),
    ]

    for label, color, i in boxes:
        x = i * 2.2
        rect = plt.Rectangle((x, 0.2), 1.8, 2.2, facecolor=color, alpha=0.85,
                               edgecolor='black', linewidth=1.5, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + 0.9, 1.3, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        if i < len(boxes) - 1:
            ax.annotate("", xy=((i + 1) * 2.2 - 0.05, 1.3),
                        xytext=(x + 1.85, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=2, color='#333'))

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("QSVT PDE Solver Pipeline", fontsize=14, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── Discrete Laplacian ──
    col_lap, col_enc = st.columns([1, 1])

    with col_lap:
        st.markdown(r"""
### Discrete Laplacian

With periodic BCs and $N = 2^n$ grid points ($\Delta x = 1/N$):

$$L_{jk} = \frac{1}{\Delta x^2}\begin{cases} -2 & j = k \\ 1 & |j-k| = 1 \bmod N \\ 0 & \text{else} \end{cases}$$

Normalized: $A = I + \frac{\nu \Delta t}{2 \Delta x^2} L$ has eigenvalues $\in [0, 1]$.

**LCU decomposition:** $A = a_0\, I + a_+\, S + a_-\, S^\dagger$

where $S$ is the QFT-based cyclic shift operator — only 3 terms!
""")

    with col_enc:
        st.markdown("#### Diffusion Matrix Structure")
        n = 8
        dx = 1.0 / n
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = -2
            A[i, (i + 1) % n] = 1
            A[(i + 1) % n, i] = 1
        A = np.eye(n) + 0.2 * A  # normalized

        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(A, cmap='RdBu_r', aspect='equal')
        ax.set_title("Normalized Laplacian (N=8)", fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    key_concept(
        "Our PDE solver <b>splits</b> the problem: QSVT handles diffusion (matrix exponential), "
        "QFT handles advection (diagonal in Fourier space). "
        "Each time step uses $O(d \\cdot n)$ gates where $d = O(\\Delta t)$ and $n = \\log_2 N$."
    )

    st.info("**Live Demo:** The 1D and 2D PDE demos are included in the upcoming slides.")

    reference_list(["Gilyen2019", "Costa2019"])
