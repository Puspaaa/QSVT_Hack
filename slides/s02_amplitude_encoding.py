"""Slide 2: Amplitude Encoding — storing classical data in quantum states."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import slide_header, reference, key_concept

TITLE = "Amplitude Encoding"


def render():
    slide_header("Amplitude Encoding",
                 "Representing classical vectors as quantum states")

    col_text, col_plot = st.columns([1.2, 1])

    with col_text:
        st.markdown(r"""
### The Idea

Given a classical vector $\vec{u} = (u_0, u_1, \dots, u_{N-1}) \in \mathbb{R}^N$, we encode it
as **amplitudes** of a quantum state:

$$
|u\rangle = \frac{1}{\|\vec{u}\|} \sum_{j=0}^{N-1} u_j\, |j\rangle
$$

where $|j\rangle$ are computational basis states of $n = \log_2 N$ qubits.

### Why This Is Powerful

| Classical | Quantum |
|-----------|---------|
| Store $N$ numbers → $O(N)$ memory | Store $N = 2^n$ amplitudes → $n$ qubits |
| Read any entry → $O(1)$ | Full state not directly readable |
| Matrix-vector product → $O(N^2)$ | Unitary application → $O(\text{poly}(n))$ |

### The Catch

Preparing an arbitrary $|u\rangle$ can itself cost $O(N)$ gates.  
But for **structured** data (e.g., PDE solutions with known physics), efficient preparation exists.
""")
        reference("NC2000")

    with col_plot:
        st.markdown("### Example: encoding a Gaussian")
        st.caption(
            "A Gaussian function discretised on 16 grid points (4 qubits) "
            "is stored as amplitudes — the histogram of measurement outcomes "
            "reproduces the function's shape."
        )

        # Build a Gaussian on 2^4 = 16 points
        n_qubits = 4
        N = 2 ** n_qubits
        x = np.linspace(-3, 3, N)
        f_x = np.exp(-x ** 2 / 2)            # unnormalised Gaussian
        norm = np.linalg.norm(f_x)
        amps = f_x / norm                     # quantum amplitudes
        probs = amps ** 2                      # Born-rule probabilities

        # Overlay plot: continuous function vs measurement histogram
        fig, ax = plt.subplots(figsize=(5, 3.2))

        # Histogram bars for measurement probabilities
        positions = np.arange(N)
        ax.bar(positions, probs, width=0.7, color="#4a90d9", alpha=0.7,
               label=r"$|\langle j|\psi\rangle|^2$ (meas. prob.)")

        # Continuous curve (rescaled to match probabilities)
        x_fine = np.linspace(0, N - 1, 200)
        f_fine = np.exp(-np.interp(x_fine, positions, x) ** 2 / 2)
        f_fine_prob = (f_fine / norm) ** 2
        ax.plot(x_fine, f_fine_prob, color="#e74c3c", lw=2.0,
                label=r"$f(x) = e^{-x^2/2}$ (scaled)")

        ax.set_xlabel("Basis state  $|j\\rangle$", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"|{j:04b}>" for j in positions],
                           fontsize=7, rotation=45, ha="right")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title("Amplitude encoding of a Gaussian", fontsize=12, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.info(
            r"**Key point:** measuring the state many times yields a histogram "
            r"whose shape matches the encoded function. "
            r"The normalisation $\|\vec{u}\|$ is tracked classically."
        )

    st.markdown("---")
    key_concept(
        "Amplitude encoding gives <b>exponential compression</b>: "
        "$2^n$ data points in $n$ qubits. "
        "The challenge shifts from <em>storage</em> to <em>state preparation</em> and <em>readout</em>."
    )
