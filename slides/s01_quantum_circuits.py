"""Slide 1: Quantum Circuits Refresher — for physics researchers."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import (
    slide_header, reference, key_concept,
    probability_bar_chart, draw_circuit_diagram, bloch_sphere,
)

TITLE = "Quantum Circuits Refresher"


# Gate definitions (name → matrix)
GATES = {
    "Hadamard (H)": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "Pauli-X": np.array([[0, 1], [1, 0]]),
    "Pauli-Z": np.array([[1, 0], [0, -1]]),
    "S gate": np.array([[1, 0], [0, 1j]]),
    "T gate": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
}


def render():
    slide_header("Quantum Circuits: A Quick Refresher",
                 "Framing quantum computation for physicists")

    st.markdown(r"""
A quantum circuit is a **sequence of unitary operators** acting on a tensor-product
Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$.  Each wire carries one qubit;
each box is a unitary gate.  Measurement projects onto the computational basis $\{|j\rangle\}$
with Born-rule probabilities $p_j = |\langle j|\psi\rangle|^2$.
""")

    st.markdown("---")

    # ── Interactive gate explorer ──
    st.markdown("### Interactive Gate Explorer")

    col_ctrl, col_matrix, col_bloch = st.columns([1, 1.5, 1.5])

    with col_ctrl:
        gate_name = st.selectbox("Choose a gate", list(GATES.keys()), key="s01_gate")
        input_state = st.radio("Input state", ["|0⟩", "|1⟩", "|+⟩"], key="s01_input",
                               horizontal=True)
        # Build input vector
        if input_state == "|0⟩":
            psi_in = np.array([1, 0], dtype=complex)
        elif input_state == "|1⟩":
            psi_in = np.array([0, 1], dtype=complex)
        else:  # |+⟩
            psi_in = np.array([1, 1], dtype=complex) / np.sqrt(2)

        U = GATES[gate_name]
        psi_out = U @ psi_in
        probs = np.abs(psi_out) ** 2

    with col_matrix:
        st.markdown(f"**Gate matrix for {gate_name}:**")
        # Display matrix nicely
        m = U
        rows = []
        for i in range(2):
            row = "  &  ".join(_format_complex(m[i, j]) for j in range(2))
            rows.append(row)
        st.latex(r"U = \begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}")

        st.markdown("**Output state:**")
        alpha_str = _format_complex(psi_out[0])
        beta_str = _format_complex(psi_out[1])
        st.latex(rf"|\psi'\rangle = {alpha_str}\,|0\rangle + {beta_str}\,|1\rangle")

        fig = probability_bar_chart(probs, ["|0⟩", "|1⟩"],
                                     title="Measurement Probabilities", figsize=(3.5, 2.5))
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col_bloch:
        st.markdown("**Bloch sphere representation:**")
        # Compute Bloch angles
        theta_b = 2 * np.arccos(min(np.abs(psi_out[0]), 1.0))
        if np.abs(psi_out[0]) > 1e-10:
            phi_b = np.angle(psi_out[1]) - np.angle(psi_out[0])
        else:
            phi_b = np.angle(psi_out[1])
        fig_bloch = bloch_sphere(theta_b, phi_b, figsize=(3.5, 3.5))
        st.pyplot(fig_bloch, use_container_width=False)
        plt.close(fig_bloch)

    st.markdown("---")

    # Key takeaway
    key_concept(
        "A quantum circuit = a <b>product of unitaries</b>. "
        "Measurement yields outcome $j$ with probability $|\\langle j|\\psi\\rangle|^2$. "
        "With $n$ qubits we work in a $2^n$-dimensional Hilbert space — "
        "this is the source of quantum advantage."
    )

    reference("NC2000")


def _format_complex(z, tol=1e-10):
    """Format a complex number for LaTeX display."""
    r, i = np.real(z), np.imag(z)
    if abs(i) < tol:
        if abs(r - round(r)) < tol:
            return str(int(round(r)))
        return f"{r:.3f}"
    if abs(r) < tol:
        if abs(i - 1) < tol:
            return "i"
        if abs(i + 1) < tol:
            return "-i"
        return f"{i:.3f}i"
    sign = "+" if i > 0 else "-"
    return f"{r:.3f}{sign}{abs(i):.3f}i"
