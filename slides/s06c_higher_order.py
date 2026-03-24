"""Slide 6c: Higher-Order Methods & Paper Comparison (Helle et al. 2025)."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from math import factorial, ceil, log2
from slides.components import slide_header, key_concept, reference_list

TITLE = "Higher-Order Methods"


def _alpha_coefficients(p: int) -> list:
    """Compute the FD coefficients αⱼ = 2(−1)^{j+1}(p!)²/((p+j)!(p−j)!) for j=1..p."""
    p_fac = factorial(p)
    coeffs = []
    for j in range(1, p + 1):
        alpha_j = 2 * ((-1) ** (j + 1)) * p_fac ** 2 / (factorial(p + j) * factorial(p - j))
        coeffs.append(alpha_j)
    return coeffs


def _draw_coefficients(p_max: int = 7):
    """Bar chart of |αⱼ| for each order p."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    p_values = [1, 2, 3, p_max]
    colors = ['#4a90d9', '#2e7d32', '#e67e22', '#7b1fa2']

    for ax, p, color in zip(axes, p_values, colors):
        alphas = _alpha_coefficients(p)
        j_vals = list(range(1, p + 1))
        ax.bar(j_vals, [abs(a) for a in alphas], color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
        ax.set_title(f"Order {2*p} (p={p})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Stencil index $j$", fontsize=9)
        ax.set_ylabel(r"$|\alpha_j|$", fontsize=9)
        ax.set_xticks(j_vals)
        ax.grid(True, alpha=0.25, axis='y')
        m = ceil(log2(2 * p + 1))
        ax.set_title(f"Order {2*p}  (m={m} ancilla)", fontsize=10, fontweight='bold', color=color)

    fig.suptitle(r"LCU Coefficients $|\alpha_j|$ for Different Finite-Difference Orders",
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def _draw_error_scaling():
    """Plot error vs grid spacing for different FD orders."""
    dx_values = np.logspace(-2, -0.3, 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {'p=1 (order 2)': '#e74c3c', 'p=2 (order 4)': '#e67e22',
              'p=3 (order 6)': '#2e7d32', 'p=7 (order 14)': '#7b1fa2'}
    for label, color in colors.items():
        p = int(label.split('=')[1].split(' ')[0])
        error = dx_values ** (2 * p)
        ax.loglog(dx_values, error, linewidth=2.5, color=color, label=label)

    ax.set_xlabel(r"Grid spacing $\Delta x$", fontsize=12)
    ax.set_ylabel(r"Error $\sim (\Delta x)^{2p}$", fontsize=12)
    ax.set_title("Error Scaling vs. Grid Spacing", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, which='both')

    # Annotate: halving Δx
    ax.annotate("Halving $\\Delta x$:\n× 4 for ord. 2\n× 64 for ord. 6",
                xy=(0.15, 0.15 ** 6), xytext=(0.035, 1e-9),
                fontsize=9, color='#333',
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

    fig.tight_layout()
    return fig


def render():
    slide_header("Higher-Order Methods & Paper Comparison",
                 "How Helle et al. 2025 improves on the standard approach")

    st.markdown(r"""
### The Key Insight: Encode the First Derivative

Our app block-encodes the **Laplacian** $\partial_x^2$ and handles advection separately (Lie-Trotter).

The paper (Helle et al. 2025) takes a different approach:

1. **Encode $H = i\beta D_{2p}$** — the first-derivative finite-difference operator (times $i$)
2. Note that $L = -cD_{2p} + \nu D_{2p}^2$ is a **polynomial in $D_{2p}$**
3. Therefore a **single QSVT call** handles both advection and diffusion simultaneously
4. Target function: $f(x; M_1, M_2) = e^{-M_1 x^2 + iM_2 x}$ where $M_1 = \nu T/\beta^2$, $M_2 = cT/\beta$

This eliminates Lie-Trotter splitting error entirely.
""")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.success(
            "**Our approach:** Lie-Trotter split\n\n"
            "$e^{t(\\nu\\nabla^2 - c\\nabla)} \\approx e^{t\\nu\\nabla^2} \\cdot e^{-tc\\nabla}$\n\n"
            "✗ First-order splitting error O(Δt²)\n\n"
            "✓ Simple to implement with existing tools"
        )
    with col2:
        st.info(
            "**Paper's approach:** Combined single call\n\n"
            "$e^{tL}$ directly via QSVT with $f(x) = e^{-M_1 x^2 + iM_2 x}$\n\n"
            "✓ No splitting error\n\n"
            "✓ One block encoding per time step"
        )

    st.markdown("---")

    # ── Higher-order finite differences ──
    st.markdown(r"""
### Higher-Order Finite-Difference Stencils

The standard (order 2) first-derivative stencil uses just two shift operators.
Higher-order stencils use more shifts but achieve much better accuracy per qubit.

The general order-$2p$ symmetric finite-difference operator is:

$$D_{2p} = \sum_{j=1}^{p} \alpha_j \delta_{2j}, \qquad \alpha_j = \frac{2(-1)^{j+1}(p!)^2}{(p+j)!\,(p-j)!}$$

where $\delta_{2j} f(x) = \frac{f(x + j\Delta x) - f(x - j\Delta x)}{2j\Delta x}$ is the $j$-th symmetric difference.
This LCU has $2p+1$ terms and needs $m = \lceil \log_2(2p+1) \rceil$ ancilla qubits.

| Order $2p$ | Ancilla $m$ | Error scaling |
|-----------|------------|--------------|
| 2 (p=1) | 2 | $O((\Delta x)^2)$ |
| 4 (p=2) | 3 | $O((\Delta x)^4)$ |
| 6 (p=3) | 3 | $O((\Delta x)^6)$ |
| 14 (p=7) | 5 | $O((\Delta x)^{14})$ |

> Order 4 and 6 **both** use only 3 ancilla qubits — the order-4 method is less efficient in this sense.
""")

    p_select = st.slider("Show stencil coefficients for orders up to p =", 1, 7, 3, 1, key="s06c_p")
    fig_coeffs = _draw_coefficients(p_max=p_select)
    st.pyplot(fig_coeffs, use_container_width=True)
    plt.close(fig_coeffs)

    st.markdown("---")

    # ── Error scaling ──
    st.markdown("### Error Scaling: Higher Order Wins")

    col_plot, col_text = st.columns([1.3, 1])
    with col_plot:
        fig_err = _draw_error_scaling()
        st.pyplot(fig_err, use_container_width=True)
        plt.close(fig_err)

    with col_text:
        st.markdown(r"""
**Rigorous bound (Theorem 7.4):**

$$\|u_T - v_T\|_{L^2} \leq T\, e^{-\nu T \mu}\, (\Delta x)^{2p}\, B$$

where $B$ depends on smoothness of $u_0$.

**Consequence:** halving $\Delta x$
- Order 2: error improves **4×**
- Order 4: error improves **16×**
- Order 6: error improves **64×**
- Order 14: error improves **16384×**

So with the same number of qubits, higher-order methods can be **orders of magnitude** more accurate.
""")

    st.markdown("---")

    # ── Paper results table ──
    st.markdown("### From the Paper: Numerical Results (Table 1 — 1D Gaussian, Pure Advection)")
    st.markdown("""
| Order | Spatial qubits | Total qubits | Error | 1-qubit gates | CNOT gates |
|-------|---------------|-------------|-------|--------------|-----------|
| **2** | 8 | 12 | 2.04 × 10⁻² | 23 433 | 16 150 |
| **2** | 9 | 13 | 5.05 × 10⁻³ | 49 298 | 33 889 |
| **6** | 6 | 11 | **1.86 × 10⁻³** | **18 658** | **13 636** |
| **6** | 7 | 12 | 3.30 × 10⁻⁵ | 37 386 | 27 130 |

*Order 6 with 6 spatial qubits (11 total) achieves lower error than order 2 with 9 spatial qubits (13 total),
while using **60% fewer CNOT gates** and **2 fewer qubits**.*

Source: Helle, Benacchio, Ousager, Andersen (arXiv:2512.22163, Table 1)
""")

    st.warning(
        "⚠️ **Why higher order isn't always better:** When the grid doesn't resolve the Fourier modes "
        "of the initial condition, higher-order methods offer no advantage — and may even perform worse. "
        "Order 14 failed for a wave-packet initial condition until the grid was fine enough to resolve "
        "all its frequency components (Paper, Section 9.1)."
    )

    key_concept(
        "Higher-order finite-difference block encodings achieve <b>exponentially better accuracy per qubit</b>. "
        "The order-6 method uses the same 3 ancilla qubits as order-4 but yields far superior accuracy. "
        "Combined with encoding the first-derivative operator, a single QSVT call replaces "
        "the split diffusion+advection circuit — eliminating Lie-Trotter splitting error entirely."
    )

    reference_list(["Helle2025", "Gilyen2019"])
