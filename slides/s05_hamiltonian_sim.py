"""Slide 5: Chebyshev Polynomials — the mathematical engine behind QSVT."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
from slides.components import slide_header, key_concept, reference_list

TITLE = "Chebyshev Polynomials"


def _draw_chebyshev_basis():
    """Plot the first few Chebyshev polynomials T_k(x) to build intuition."""
    x = np.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = ['#1565c0', '#e65100', '#2e7d32', '#7b1fa2', '#c62828']
    for k in range(5):
        coeffs = np.zeros(k + 1)
        coeffs[k] = 1.0
        y = chebval(x, coeffs)
        ax.plot(x, y, color=colors[k], linewidth=2.2 if k < 3 else 1.5,
                label=f'$T_{k}(x)$', alpha=0.9)
    ax.axhline(1, color='grey', lw=0.6, ls='--', alpha=0.5)
    ax.axhline(-1, color='grey', lw=0.6, ls='--', alpha=0.5)
    ax.axhline(0, color='grey', lw=0.4, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$T_k(x)$", fontsize=12)
    ax.set_title("Chebyshev Polynomials of the First Kind", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, ncol=5, loc='lower center', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _draw_interactive_approx(target_name: str, degree: int):
    """Interactive Chebyshev approximation plot with top panel + error panel."""
    x = np.linspace(-1, 1, 800)

    # Target definitions  ─────────────────────────────────────────────────
    if target_name == "sign(x)":
        # smooth-ish sign for fitting; true sign for display
        y_target = np.sign(x)
        # fit via odd Chebyshev (sign is odd)
        n_fit = 600
        x_fit = np.cos(np.pi * np.arange(n_fit) / (n_fit - 1))
        y_fit = np.sign(x_fit)
        coeffs = np.polynomial.chebyshev.chebfit(x_fit, y_fit, degree)
        # enforce odd parity (zero even coeffs)
        for i in range(len(coeffs)):
            if i % 2 == 0:
                coeffs[i] = 0.0
        parity_label = "odd"
        title = f"sign$(x)$ — Chebyshev degree {degree} ({parity_label})"
        ylim = (-1.45, 1.45)
    elif target_name == "step(x − 0.3)":
        y_target = np.where(x > 0.3, 1.0, 0.0).astype(float)
        n_fit = 600
        x_fit = np.cos(np.pi * np.arange(n_fit) / (n_fit - 1))
        y_fit = np.where(x_fit > 0.3, 1.0, 0.0).astype(float)
        coeffs = np.polynomial.chebyshev.chebfit(x_fit, y_fit, degree)
        parity_label = "full"
        title = f"step$(x - 0.3)$ — Chebyshev degree {degree}"
        ylim = (-0.5, 1.5)
    else:  # 1/x (bounded)
        kappa = 0.1
        y_target = np.where(np.abs(x) > kappa,
                            kappa / x, np.sign(x) * 1.0).astype(float)
        n_fit = 600
        x_fit = np.cos(np.pi * np.arange(n_fit) / (n_fit - 1))
        y_fit_raw = np.where(np.abs(x_fit) > kappa,
                             kappa / x_fit, np.sign(x_fit) * 1.0).astype(float)
        coeffs = np.polynomial.chebyshev.chebfit(x_fit, y_fit_raw, degree)
        # enforce odd parity (1/x is odd)
        for i in range(len(coeffs)):
            if i % 2 == 0:
                coeffs[i] = 0.0
        parity_label = "odd"
        title = f"$\\kappa / x$ (bounded, $\\kappa={kappa}$) — Chebyshev degree {degree} ({parity_label})"
        ylim = (-1.45, 1.45)

    y_approx = chebval(x, coeffs)
    error = np.abs(y_target - y_approx)
    max_err = np.max(error)

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), height_ratios=[3, 1],
                                     gridspec_kw={'hspace': 0.08})

    ax1.plot(x, y_target, 'b-', linewidth=2.2, label='Target $f(x)$', zorder=2)
    ax1.plot(x, y_approx, 'r--', linewidth=2, label=f'Chebyshev deg-{degree}', zorder=3)
    ax1.fill_between(x, y_target, y_approx, alpha=0.12, color='red', zorder=1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(ylim)
    ax1.set_ylabel("$f(x)$", fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.2)

    # Highlight Gibbs overshoots
    overshoot = np.abs(y_approx) > 1.0
    if np.any(overshoot):
        ax1.fill_between(x, -1, 1, where=overshoot, alpha=0.08,
                         color='orange', label='> QSVT bound')
        ax1.axhline(1, color='#c62828', lw=0.8, ls='--', alpha=0.6)
        ax1.axhline(-1, color='#c62828', lw=0.8, ls='--', alpha=0.6)

    ax2.semilogy(x, error + 1e-16, 'green', linewidth=1.5)
    ax2.set_xlabel("$x$", fontsize=12)
    ax2.set_ylabel("|Error|", fontsize=11)
    ax2.set_xlim(-1, 1)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig, max_err


def render():
    slide_header("Chebyshev Polynomials",
                 "The mathematical backbone of QSVT polynomial design")

    # ── Why Chebyshev? ──
    col_text, col_plot = st.columns([1, 1.2])

    with col_text:
        st.markdown(r"""
### Why Chebyshev Polynomials?

QSVT can implement any polynomial $P$ with $|P(x)| \leq 1$ on $[-1,1]$.
But how do we **find** such a polynomial that approximates the function we want?

**Chebyshev polynomials** $T_k(x) = \cos(k \arccos x)$ are the natural choice:

- **Bounded:** $|T_k(x)| \leq 1$ on $[-1, 1]$ — satisfies QSVT's constraint automatically
- **Optimal:** best $L^\infty$ (minimax) approximation among polynomials of the same degree
- **Definite parity:** $T_k(-x) = (-1)^k T_k(x)$ — even/odd decomposition is trivial
- **Fast convergence:** for smooth functions, coefficients decay exponentially
""")

    with col_plot:
        fig_basis = _draw_chebyshev_basis()
        st.pyplot(fig_basis, use_container_width=True)
        plt.close(fig_basis)

    st.markdown("---")

    # ── The parity constraint ──
    st.markdown(r"""
### The Parity Constraint

QSVT requires that $P$ has **definite parity**: either $P(-x) = P(x)$ (even) or $P(-x) = -P(x)$ (odd).

This is not a serious limitation — any function can be split:

$$
f(x) = \underbrace{\frac{f(x) + f(-x)}{2}}_{\text{even part}} + \underbrace{\frac{f(x) - f(-x)}{2}}_{\text{odd part}}
$$

Each part is implemented by a separate QSVT circuit, then combined.

**Example — Hamiltonian simulation:** We want $e^{itx}$, which has no definite parity.  
Instead we target the even function $e^{t(|x| - 1)}$, which agrees with $e^{t(x-1)}$ for $x \geq 0$
and is naturally even.
""")

    st.markdown("---")

    # ── Interactive approximation demo ──
    st.markdown("### Convergence: more terms → better approximation")

    col_ctrl, col_plot = st.columns([1, 2.2])

    with col_ctrl:
        target_name = st.selectbox(
            "Target function",
            ["sign(x)", "step(x − 0.3)", "1/x (bounded)"],
            key="s05_target",
        )
        cheb_deg = st.slider("Chebyshev degree $d$", 3, 61, 11, 2, key="s05_deg")

        if target_name == "sign(x)":
            st.markdown(r"""
The **sign** function is the classic hard case:
- Discontinuity at $x = 0$ causes **Gibbs oscillations**
- Higher degree → more wiggles near the jump
- Odd parity enforced ($P(-x) = -P(x)$)
- Used in eigenvalue thresholding & phase estimation
            """)
        elif target_name == "step(x − 0.3)":
            st.markdown(r"""
The **step** function (threshold at $0.3$):
- Jump discontinuity → Gibbs phenomenon
- No definite parity → would need even + odd QSVT circuits
- Used for projecting onto eigenvalue windows
            """)
        else:
            st.markdown(r"""
**Matrix inversion** ($1/x$, bounded away from 0):
- Pole at $x=0$ → truncated to $|\kappa/x| \leq 1$
- Odd parity enforced
- This is the polynomial behind HHL
- Condition number $\kappa$ controls difficulty
            """)

    with col_plot:
        fig_approx, max_err = _draw_interactive_approx(target_name, cheb_deg)
        st.pyplot(fig_approx, use_container_width=True)
        plt.close(fig_approx)
        st.metric("Max approximation error", f"{max_err:.2e}")

    st.markdown("---")

    # ── From function to QSVT angles ──
    st.markdown(r"""
### From Target Function to QSVT Angles

The full pipeline to turn a classical function into a quantum circuit:

1. **Choose target function** $f(x)$ (e.g. $e^{-itx}$, $1/x$, sign, step)
2. **Chebyshev approximation** → polynomial $P(x) = \sum_k c_k\, T_k(x)$ with $|P| \leq 1$
3. **Enforce parity** → split into even/odd parts if needed
4. **Compute QSVT angles** $\{\phi_j\}$ (classical optimisation, e.g. `pyqsp`)
5. **Build circuit** → interleave $U_A$ with $e^{i\phi_j Z}$ rotations

The **degree** $d$ of the polynomial directly gives the **query complexity**: each extra term
costs one additional call to the block encoding.
""")

    key_concept(
        "Chebyshev polynomials are the <b>design language</b> for QSVT. "
        "They naturally satisfy the boundedness and parity constraints, converge rapidly, "
        "and their degree directly determines the quantum circuit depth. "
        "Classical preprocessing computes the QSVT angles from the Chebyshev coefficients."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017", "Low2019"])
