"""Slide 3: Block Encoding & LCU — the interface between classical matrices and quantum circuits."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import (
    slide_header, reference, key_concept, matrix_heatmap, reference_list,
)

TITLE = "Block Encoding & LCU"


def render():
    slide_header("Block Encoding & LCU",
                 "Embedding classical matrices inside unitaries")

    # ── Definition ──
    st.markdown(r"""
### What Is a Block Encoding?

A unitary $U_A$ acting on $a + n$ qubits is an **$(\alpha, a)$-block encoding** of an
$N \times N$ matrix $A$ if:

$$
A = \alpha\, (\langle 0|^{\otimes a} \otimes I_N)\; U_A\; (|0\rangle^{\otimes a} \otimes I_N)
$$

In other words, $A/\alpha$ sits in the **top-left block** of $U_A$.  
When we prepare $|0\rangle^{\otimes a}|\psi\rangle$, apply $U_A$, and **postselect** the ancilla on $|0\rangle^{\otimes a}$,
we effectively apply $A/\alpha$ to $|\psi\rangle$.
""")

    # ── Visual ──
    col_vis, col_demo = st.columns([1, 1])

    with col_vis:
        st.markdown("#### Block Structure of $U_A$")

        # Build a visual block matrix
        fig, ax = plt.subplots(figsize=(4, 4))
        # Full unitary background
        rect_full = plt.Rectangle((0, 0), 4, 4, facecolor='#e0e0e0',
                                   edgecolor='black', linewidth=2)
        ax.add_patch(rect_full)
        # A/alpha block (top-left)
        rect_a = plt.Rectangle((0, 2), 2, 2, facecolor='#4a90d9', alpha=0.85,
                                edgecolor='black', linewidth=2)
        ax.add_patch(rect_a)
        ax.text(1, 3, r"$A/\alpha$", fontsize=18, ha='center', va='center',
                color='white', fontweight='bold')
        ax.text(3, 3, r"$\cdot$", fontsize=18, ha='center', va='center', color='#666')
        ax.text(1, 1, r"$\cdot$", fontsize=18, ha='center', va='center', color='#666')
        ax.text(3, 1, r"$\cdot$", fontsize=18, ha='center', va='center', color='#666')

        ax.text(1, 4.3, "ancilla = |0⟩", fontsize=10, ha='center', color='#4a90d9')
        ax.text(3, 4.3, "ancilla ≠ |0⟩", fontsize=10, ha='center', color='#999')

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Unitary $U_A$", fontsize=13, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_demo:
        st.markdown("#### Interactive: 2×2 Block Encoding")
        st.caption("Adjust the parameter to see how $A$ embeds inside $U_A$.")

        a_param = st.slider("Matrix parameter $a$", 0.0, 1.0, 0.5, 0.05, key="s03_a")

        # Simple 2x2 A = a * I (trivially block-encoded)
        A = np.array([[a_param, 0], [0, a_param]])

        # Build a 4x4 unitary block encoding: U = [[A, sqrt(I-A^2)], [sqrt(I-A^2), -A]]
        sqrtc = np.sqrt(max(1 - a_param ** 2, 0))
        U = np.array([
            [a_param, 0, sqrtc, 0],
            [0, a_param, 0, sqrtc],
            [sqrtc, 0, -a_param, 0],
            [0, sqrtc, 0, -a_param],
        ])

        fig = matrix_heatmap(U, title=f"$U_A$ (4×4 unitary, a={a_param:.2f})", figsize=(4.5, 3.5))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            f"Top-left 2×2 block = $A = {a_param:.2f}\\cdot I_2$.  "
            f"Postselection success probability = $a^2 = {a_param**2:.3f}$."
        )

    st.markdown("---")

    # ── LCU ──
    st.markdown(r"""
### Linear Combination of Unitaries (LCU)

If we can decompose $A = \sum_{i=0}^{m-1} \alpha_i\, U_i$ where each $U_i$ is easy to implement, then:

1. **PREPARE** oracle: $|0\rangle \mapsto \sum_i \sqrt{\alpha_i / s}\; |i\rangle$ where $s = \sum |\alpha_i|$
2. **SELECT** oracle: $|i\rangle|\psi\rangle \mapsto |i\rangle\, U_i|\psi\rangle$
3. **Block encoding**: $U_A = \text{PREPARE}^\dagger \cdot \text{SELECT} \cdot \text{PREPARE}$

#### In Our Diffusion Operator

$$A_{\text{diff}} = a_0\, I + a_+\, S + a_-\, S^\dagger$$

where $S$ is the QFT-based cyclic shift. This uses just **2 ancilla qubits** and 3 unitaries.
""")

    key_concept(
        "Block encoding is the <b>interface</b> between classical linear algebra and quantum circuits. "
        "LCU is the most common construction: decompose $A$ into a sum of easy unitaries."
    )

    reference_list(["Gilyen2019", "Berry2015"])
