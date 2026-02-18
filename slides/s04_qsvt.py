"""Slide 4: QSVT — The Grand Unification."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval, chebpts2
from slides.components import (
    slide_header, reference, key_concept, reference_list,
    qsvt_circuit_schematic, polynomial_plot,
)

TITLE = "QSVT: The Grand Unification"


def render():
    slide_header("QSVT: The Grand Unification",
                 "One framework to rule (almost) all quantum algorithms")

    st.markdown(r"""
### The Central Theorem

Given an $(\alpha, a)$-block encoding $U_A$ of matrix $A$ with singular values $\{\sigma_i\}$,
and a polynomial $P$ of degree $d$ satisfying:

1. $|P(x)| \leq 1$ for all $x \in [-1, 1]$
2. $P$ has **definite parity** (even or odd)

then there exist angles $\{\phi_0, \phi_1, \dots, \phi_d\}$ such that the QSVT circuit
block-encodes $P(A/\alpha)$:

$$
(\langle 0| \otimes I)\;\left[e^{i\phi_0 Z}\prod_{j=1}^{d} U_A^{(\dagger)}\; e^{i\phi_j Z}\right]\;(|0\rangle \otimes I) = P(A/\alpha)
$$

This applies $P$ to each **singular value**: $\sigma_i \mapsto P(\sigma_i)$.
""")

    st.markdown("---")

    # ── Circuit diagram ──
    st.markdown("### QSVT Circuit Structure")

    degree = st.slider("Polynomial degree $d$", 2, 20, 6, 1, key="s04_degree")

    fig_circ = qsvt_circuit_schematic(degree, figsize=(14, 2.5))
    st.pyplot(fig_circ, use_container_width=True)
    plt.close(fig_circ)

    st.markdown(r"""
The circuit alternates $U_A$ and $U_A^\dagger$ (giving access to both singular value signs),
interleaved with **signal rotations** $e^{i\phi_j Z}$ on a single ancilla qubit.
The total query complexity is $O(d)$ — exactly the degree of the polynomial.
""")

    st.markdown("---")

    # ── Chebyshev approximation demo ──
    st.markdown("### Chebyshev Polynomial Approximation")

    col_fn, col_plot = st.columns([1, 2])

    with col_fn:
        target_choice = st.selectbox(
            "Target function",
            ["exp(t(|x|-1))", "sign(x)", "step(x-0.3)", "1/x (bounded)"],
            key="s04_target",
        )
        cheb_deg = st.slider("Approximation degree", 2, 40, 12, 2, key="s04_cheb_deg")
        
        st.markdown(r"""
        **Chebyshev basis:**
        $$P(x) = \sum_{k=0}^{d} c_k\, T_k(x)$$
        where $T_k(\cos\theta) = \cos(k\theta)$.
        
        Key properties:
        - Optimal $L^\infty$ approximation
        - Satisfy $|T_k(x)| \leq 1$ on $[-1,1]$
        - Definite parity: $T_k(-x) = (-1)^k T_k(x)$
        """)

    with col_plot:
        # Build target function
        if target_choice == "exp(t(|x|-1))":
            t_param = st.slider("Parameter $t$", 0.1, 5.0, 1.0, 0.1, key="s04_t")
            target_fn = lambda x: np.exp(t_param * (np.abs(x) - 1))
            title = f"$e^{{{t_param:.1f}(|x|-1)}}$ — Chebyshev degree {cheb_deg}"
        elif target_choice == "sign(x)":
            target_fn = lambda x: np.sign(x) if abs(x) > 0.05 else 20 * x
            title = f"sign$(x)$ — Chebyshev degree {cheb_deg}"
        elif target_choice == "step(x-0.3)":
            target_fn = lambda x: 1.0 if x > 0.3 else 0.0
            title = f"step$(x-0.3)$ — Chebyshev degree {cheb_deg}"
        else:  # 1/x bounded
            target_fn = lambda x: 1.0 / max(abs(x), 0.1) * np.sign(x) * 0.1
            title = f"$0.1/x$ (bounded) — Chebyshev degree {cheb_deg}"

        # Least-squares Chebyshev fit
        n_sample = 200
        x_sample = np.cos(np.pi * np.arange(n_sample) / (n_sample - 1))  # Chebyshev nodes
        y_sample = np.array([target_fn(xi) for xi in x_sample])
        coeffs = np.polynomial.chebyshev.chebfit(x_sample, y_sample, cheb_deg)

        fig = polynomial_plot(
            target_fn, coeffs, domain=(-1, 1),
            target_label="Target $f(x)$",
            approx_label=f"Chebyshev deg-{cheb_deg}",
            title=title,
            figsize=(7, 4.5),
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # ── Grand unification table ──
    st.markdown("### Algorithms Unified by QSVT")

    st.markdown(r"""
| Algorithm | Polynomial $P(\sigma)$ | Degree | Application |
|-----------|----------------------|--------|-------------|
| **Hamiltonian simulation** | $e^{-it\sigma}$ | $O(t + \log(1/\epsilon))$ | Time evolution |
| **Matrix inversion (HHL)** | $1/\sigma$ | $O(\kappa/\epsilon)$ | Linear systems |
| **Amplitude amplification** | Chebyshev of $\sigma$ | $O(1/\sqrt{p})$ | Search |
| **Phase estimation** | Step function | $O(1/\epsilon)$ | Eigenvalues |
| **Quantum walks** | $\sigma \mapsto e^{i\arccos\sigma}$ | $O(1)$ | Graph problems |
""")

    key_concept(
        "QSVT is a <b>grand unification</b>: most quantum speedups reduce to applying a polynomial "
        "to singular values of a block-encoded matrix. The polynomial determines the algorithm; "
        "the angles $\\{\\phi_j\\}$ are found by classical preprocessing."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017"])
