"""Slide 2: Amplitude Encoding — storing classical data in quantum states."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import slide_header, reference, key_concept, probability_bar_chart

TITLE = "Amplitude Encoding"


def render():
    slide_header("Amplitude Encoding",
                 "Representing classical vectors as quantum states")

    col_text, col_demo = st.columns([1.2, 1])

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

    with col_demo:
        st.markdown("### Interactive Demo")
        st.caption("Drag the sliders to define a 4-component vector, see the quantum state.")

        v0 = st.slider("$u_0$", -2.0, 2.0, 1.0, 0.1, key="s02_v0")
        v1 = st.slider("$u_1$", -2.0, 2.0, 0.5, 0.1, key="s02_v1")
        v2 = st.slider("$u_2$", -2.0, 2.0, -0.3, 0.1, key="s02_v2")
        v3 = st.slider("$u_3$", -2.0, 2.0, 0.8, 0.1, key="s02_v3")

        vec = np.array([v0, v1, v2, v3])
        norm = np.linalg.norm(vec)

        if norm < 1e-8:
            st.warning("Vector is zero — move at least one slider.")
        else:
            normed = vec / norm
            probs = normed ** 2

            st.latex(
                r"|\vec{u}| = " + f"{norm:.3f}" +
                r", \quad |u\rangle = " +
                " + ".join(f"{normed[i]:+.3f}\\,|{i}\\rangle" for i in range(4))
            )

            fig = probability_bar_chart(
                probs,
                ["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
                title="Measurement Probabilities (2 qubits)",
                figsize=(5, 3),
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.info(
                f"**Normalization:** $\\|\\vec{{u}}\\| = {norm:.3f}$  \n"
                f"This information is lost in the quantum state and must be tracked classically."
            )

    st.markdown("---")
    key_concept(
        "Amplitude encoding gives <b>exponential compression</b>: "
        "$2^n$ data points in $n$ qubits. "
        "The challenge shifts from <em>storage</em> to <em>state preparation</em> and <em>readout</em>."
    )
