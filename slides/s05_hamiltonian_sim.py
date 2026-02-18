"""Slide 5: Hamiltonian Simulation via QSVT."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import slide_header, reference, key_concept, reference_list

TITLE = "Hamiltonian Simulation"


def render():
    slide_header("Hamiltonian Simulation via QSVT",
                 "Computing $e^{tA}$ as a polynomial of singular values")

    col_theory, col_demo = st.columns([1.2, 1])

    with col_theory:
        st.markdown(r"""
### From Physics to Polynomials

Many physics problems reduce to computing the **matrix exponential**:

$$\frac{\partial}{\partial t} |\psi\rangle = A\,|\psi\rangle \quad\Longrightarrow\quad |\psi(t)\rangle = e^{tA}\,|\psi(0)\rangle$$

If we have a block encoding of $A$ (with eigenvalues in $[-1, 1]$), we need a polynomial $P(x) \approx e^{tx}$.

### The Parity Constraint

QSVT requires **definite parity**: $P(-x) = \pm P(x)$.  
But $e^{tx}$ has no definite parity!

**Solution:** Target the **even function**:

$$\tilde{P}(x) = e^{t(|x| - 1)}$$

This satisfies $\tilde{P}(-x) = \tilde{P}(x)$ (even parity) and agrees with $e^{t(x-1)}$ for $x \geq 0$.

### Complexity

The Chebyshev degree needed: $d \sim O(t)$ for accuracy $\epsilon$.  
Compared to classical dense exponentiation: $O(N^3)$ — quantum: $O(t \cdot \text{poly}(\log N))$.
""")

    with col_demo:
        st.markdown("### Interactive: Target Function")

        t_val = st.slider("Evolution time $t$", 0.1, 10.0, 2.0, 0.1, key="s05_t")
        deg_val = st.slider("Chebyshev degree $d$", 2, 50, 12, 2, key="s05_deg")

        x = np.linspace(-1, 1, 500)
        y_target = np.exp(t_val * (np.abs(x) - 1))

        # Chebyshev fit (even terms only for even parity)
        n_fit = 300
        x_fit = np.cos(np.pi * np.arange(n_fit) / (n_fit - 1))
        y_fit = np.exp(t_val * (np.abs(x_fit) - 1))
        coeffs_all = np.polynomial.chebyshev.chebfit(x_fit, y_fit, deg_val)
        # Zero out odd coefficients to enforce even parity
        for i in range(len(coeffs_all)):
            if i % 2 == 1:
                coeffs_all[i] = 0.0

        from numpy.polynomial.chebyshev import chebval
        y_approx = chebval(x, coeffs_all)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), height_ratios=[3, 1],
                                         gridspec_kw={'hspace': 0.08})

        ax1.plot(x, y_target, 'b-', linewidth=2, label=r'Target $e^{t(|x|-1)}$')
        ax1.plot(x, y_approx, 'r--', linewidth=2, label=f'Even Chebyshev (deg {deg_val})')
        ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_ylabel("f(x)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_title(f"Hamiltonian Evolution: $t = {t_val:.1f}$, degree = {deg_val}",
                       fontsize=13, fontweight='bold')
        ax1.set_xlim(-1, 1)
        ax1.tick_params(labelbottom=False)
        ax1.grid(True, alpha=0.3)

        error = np.abs(y_target - y_approx)
        ax2.semilogy(x, error + 1e-16, 'green', linewidth=1.5)
        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("|Error|", fontsize=11)
        ax2.set_xlim(-1, 1)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        max_err = np.max(error)
        st.metric("Max approximation error", f"{max_err:.2e}")

        if max_err > 0.05:
            st.warning(f"Increase degree for better accuracy (try d ≥ {int(t_val * 4)})")

    st.markdown("---")
    key_concept(
        "Hamiltonian simulation via QSVT: approximate $e^{t(|x|-1)}$ with an <b>even Chebyshev polynomial</b> "
        "of degree $d = O(t)$. The circuit uses $d$ queries to the block encoding — "
        "<b>optimal</b> in query complexity."
    )

    reference_list(["Low2017", "Low2019"])
