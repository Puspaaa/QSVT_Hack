"""Slide 7: QSVT for Numerical Integration."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import slide_header, reference, key_concept, reference_list

TITLE = "Quantum Integration"


def render():
    slide_header("Quantum Numerical Integration",
                 "Computing $\\int_a^b f(x)\\,dx$ via quantum overlap estimation")

    st.markdown(r"""
### Problem Statement

Given a function $f(x)$ encoded as a quantum state, estimate:

$$I = \int_a^b f(x)\,dx$$

### Quantum Formulation

Discretize on $N = 2^n$ grid points: $x_j = j/N$.

1. Encode $f$ as amplitudes: $|u\rangle = \frac{1}{\|u\|} \sum_j f(x_j)\,|j\rangle$
2. Encode the domain indicator: $|D\rangle = \frac{1}{\sqrt{|D|}} \sum_{j \in [a,b]} |j\rangle$
3. The integral becomes an **inner product**:

$$I \approx \langle D\,|\,u\rangle \;\cdot\; \frac{\sqrt{|D|}\;\|u\|}{N}$$
""")

    st.markdown("---")

    # ── Three methods ──
    st.markdown("### Three Methods Implemented")

    tab1, tab2, tab3 = st.tabs(["Method 1: Compute-Uncompute", "Method 2: Arithmetic", "Method 3: QSVT"])

    with tab1:
        st.markdown(r"""
#### Compute-Uncompute (Swap Test)

**When:** Integration domain aligns with qubit structure ([0, 0.5], [0.25, 0.75], etc.)

**Circuit:** Prepare $|D\rangle$ with $O(n)$ Hadamard gates, apply $U_f^\dagger$, measure probability of $|0^n\rangle$.

$$P(0^n) = |\langle D | u \rangle|^2$$

**Complexity:** $O(n)$ depth — the lightest method.

**Limitation:** Only works for intervals that align with binary partitions.
""")

    with tab2:
        st.markdown(r"""
#### Arithmetic Comparator

**When:** Arbitrary interval $[a, b]$.

**Idea:** Build a quantum **range oracle** using integer comparators:
- $O_D|j\rangle = (-1)^{[j \in D]}|j\rangle$ where $D = \{j : a \leq x_j \leq b\}$

Uses Qiskit's `IntegerComparator` circuits (ripple-carry arithmetic).

**Optional:** Apply **Grover amplitude amplification** to boost the overlap signal.

**Complexity:** $O(Mn)$ with $M$ Grover iterations, each using $O(n)$ gates.
""")
        reference("Brassard2002")

    with tab3:
        st.markdown(r"""
#### QSVT Parity Decomposition

**When:** Arbitrary interval — the most general method.

**Idea:** Approximate the **boxcar indicator** $\mathbb{1}_{[a,b]}$ as a polynomial in the
eigenvalue $\lambda = \cos(\pi x)$ of a block-encoded cosine operator.

$$\mathbb{1}_{[a,b]}(x) \approx P_{\text{even}}(\lambda) + P_{\text{odd}}(\lambda)$$

Two QSVT circuits (one for each parity component), combined to estimate the overlap.

**Complexity:** $O(d \cdot n)$ where $d$ is the polynomial degree (controls precision of the boxcar).
""")
        reference("Gilyen2019")

    st.markdown("---")

    # ── Boxcar visualization ──
    st.markdown("### Interactive: Boxcar Polynomial Approximation")

    col1, col2 = st.columns([1, 2])

    with col1:
        a_val = st.slider("Interval start $a$", 0.0, 0.9, 0.2, 0.05, key="s07_a")
        b_val = st.slider("Interval end $b$", a_val + 0.05, 1.0, 0.7, 0.05, key="s07_b")
        box_deg = st.slider("Polynomial degree", 4, 60, 20, 2, key="s07_deg")

    with col2:
        x = np.linspace(0, 1, 500)
        lam = np.cos(np.pi * x)  # eigenvalue mapping

        # True boxcar
        boxcar = np.where((x >= a_val) & (x <= b_val), 1.0, 0.0)

        # Polynomial approximation of boxcar in lambda-space
        # Use smoothed boxcar for Chebyshev fit
        lam_sample = np.cos(np.pi * np.linspace(0, 1, 400))
        box_sample = np.where(
            (np.linspace(0, 1, 400) >= a_val) & (np.linspace(0, 1, 400) <= b_val),
            1.0, 0.0
        )
        coeffs = np.polynomial.chebyshev.chebfit(lam_sample, box_sample, box_deg)
        from numpy.polynomial.chebyshev import chebval
        box_approx = chebval(lam, coeffs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Position space
        ax1.fill_between(x, 0, boxcar, alpha=0.3, color='blue', label='True indicator')
        ax1.plot(x, np.clip(box_approx, -0.2, 1.2), 'r-', linewidth=2, 
                 label=f'Polynomial (deg {box_deg})')
        ax1.set_xlabel("$x$", fontsize=12)
        ax1.set_ylabel("$\\mathbb{1}_{[a,b]}(x)$", fontsize=12)
        ax1.set_title("Position Space", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.set_ylim(-0.2, 1.3)
        ax1.grid(True, alpha=0.3)

        # Eigenvalue space
        ax2.plot(lam, boxcar, 'b-', linewidth=2, alpha=0.5, label='True indicator')
        ax2.plot(lam, np.clip(box_approx, -0.2, 1.2), 'r-', linewidth=2,
                 label=f'$P(\\lambda)$ deg {box_deg}')
        ax2.set_xlabel("$\\lambda = \\cos(\\pi x)$", fontsize=12)
        ax2.set_title("Eigenvalue Space", fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.set_ylim(-0.2, 1.3)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # ── Comparison table ──
    st.markdown("### Method Comparison")
    st.markdown(r"""
| | **Method 1** | **Method 2** | **Method 3** |
|---|---|---|---|
| **Interval** | Binary-aligned only | Arbitrary | Arbitrary |
| **Qubits** | $n$ | $n + O(n)$ ancilla | $n + 2$ |
| **Depth** | $O(n)$ | $O(Mn)$ | $O(dn)$ |
| **Strengths** | Simplest, shallowest | Exact range, Grover boost | Most general, QSVT-native |
| **Weaknesses** | Limited intervals | Many ancilla qubits | Polynomial approximation error |
""")

    st.info("**Live Demo:** The integration demo is included in the upcoming slides.")

    key_concept(
        "Quantum integration reformulates $\\int_a^b f(x)dx$ as an <b>inner product</b> "
        "$\\langle D|u\\rangle$ — a naturally quantum quantity. "
        "QSVT approximates the domain indicator as a polynomial, enabling arbitrary intervals."
    )

    reference_list(["Gilyen2019", "Brassard2002"])
