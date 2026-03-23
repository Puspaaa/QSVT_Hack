"""Slide 5: Chebyshev Polynomials — the mathematical engine behind QSVT."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
from scipy.special import jv as bessel_j
from slides.components import slide_header, key_concept, reference_list, reference

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
        sigma_min = 0.1
        y_target = np.where(np.abs(x) > sigma_min,
                            sigma_min / x, np.sign(x) * 1.0).astype(float)
        n_fit = 600
        x_fit = np.cos(np.pi * np.arange(n_fit) / (n_fit - 1))
        y_fit_raw = np.where(np.abs(x_fit) > sigma_min,
                             sigma_min / x_fit, np.sign(x_fit) * 1.0).astype(float)
        coeffs = np.polynomial.chebyshev.chebfit(x_fit, y_fit_raw, degree)
        # enforce odd parity (1/x is odd)
        for i in range(len(coeffs)):
            if i % 2 == 0:
                coeffs[i] = 0.0
        parity_label = "odd"
        title = (
            f"$\\sigma_{{\\min}} / x$ (bounded, $\\sigma_{{\\min}}={sigma_min}$) "
            f"— Chebyshev degree {degree} ({parity_label})"
        )
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

Why do these constraints appear at all?

- **Boundedness ($|P(x)|\le 1$):** the QSVT circuit is unitary, and the implemented block is a sub-block of that unitary after projection.
    So its singular values cannot exceed 1. If your target transform is larger than 1 on part of the spectrum,
    you must rescale it and keep track of that scaling separately.

- **Definite parity:** the alternating QSP/QSVT sequence built from $U_A$ and $U_A^\dagger$ with phase rotations has a built-in symmetry.
    That symmetry makes a single sequence implement either an even or an odd polynomial channel.
    A generic function is therefore split into even and odd parts and synthesized via two channels.

This is not a serious limitation — any function can be split:

$$
f(x) = \underbrace{\frac{f(x) + f(-x)}{2}}_{\text{even part}} + \underbrace{\frac{f(x) - f(-x)}{2}}_{\text{odd part}}
$$

Each part is implemented by a separate QSVT circuit, then combined.

**Example — Hamiltonian simulation:** We want $e^{iMx} = \cos(Mx) + i\sin(Mx)$.
- Even part: $\cos(Mx)$ → one QSVT circuit
- Odd part: $\sin(Mx)$ → a second QSVT circuit
- Both have rigorous Chebyshev expansions via the **Jacobi-Anger formula** (see below).

**Example — Diffusion:** We want $e^{-Mx^2}$ — this is naturally **even**, so only one circuit is needed.
Our PDE solver targets $e^{t(|x|-1)}$ (a closely related even decay function) for the discrete diffusion step.

**Combined (Helle et al. 2025):** The paper encodes the first-derivative operator $H = i\beta D_{2p}$ and targets
$f(x; M_1, M_2) = e^{-M_1 x^2 + iM_2 x}$ — handling diffusion and advection in a *single* QSVT call,
avoiding any Lie-Trotter splitting error.
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
- Pole at $x=0$ → truncated to $|\kappa/x| \leq 1$ where $\kappa = \sigma_{\min}/\alpha$
- Odd parity enforced
- This is the polynomial behind HHL
- Condition number $\kappa_{\mathrm{cond}} = 1/\kappa = \alpha/\sigma_{\min}$ controls difficulty: larger gap $\kappa$ → lower degree needed
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

    st.markdown("---")

    # ── Jacobi-Anger expansion ──
    st.markdown("### Analytic Polynomial Construction: Jacobi-Anger Expansion")

    st.markdown(r"""
For numerical fitting (our demos above), we use `numpy.chebfit`. But for rigorous degree bounds
and guaranteed error control, the **Jacobi-Anger expansion** gives analytic Chebyshev coefficients.

**Key formula:**
$$e^{iMx} = J_0(M) + 2\sum_{n=1}^{\infty} i^n\, J_n(M)\, T_n(x), \qquad x \in [-1, 1]$$

where $J_n(M)$ are **Bessel functions of the first kind**.
Separating real and imaginary parts:
$$\cos(Mx) = J_0(M) + 2\sum_{k=1}^{\infty}(-1)^k J_{2k}(M)\, T_{2k}(x) \quad\text{(even)}$$
$$\sin(Mx) = 2\sum_{k=0}^{\infty}(-1)^k J_{2k+1}(M)\, T_{2k+1}(x) \quad\text{(odd)}$$

**Why this matters:** Bessel functions decay rapidly for $n > M$, so truncating at degree $R \approx eM/2$ gives
error $\varepsilon$. The rigorous bound is $R = \lfloor r(eM/2,\, 5\varepsilon/4) \rfloor$ where $r(t,\varepsilon)$ satisfies $(t/r)^r = \varepsilon$.
""")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        M_val = st.slider("Parameter $M$ (frequency)", 1.0, 20.0, 5.0, 0.5, key="s05_ja_M")
        R_trunc = st.slider("Truncation degree $R$", 2, 60, 10, 1, key="s05_ja_R")
        show_bessel = st.checkbox("Show Bessel coefficients", value=True, key="s05_ja_bessel")

        # Rigorous degree estimate
        # r(t, eps) where t = e*M/2, eps = 0.01: use R ~ ceil(e*M/2) as rough guide
        R_rigorous = int(np.ceil(np.e * M_val / 2))
        st.metric("Rigorous min. degree (ε=0.01)", f"R ≈ {R_rigorous}")
        if R_trunc < R_rigorous:
            st.warning(f"R={R_trunc} < R_rigorous={R_rigorous}: expect visible error")
        else:
            st.success(f"R={R_trunc} ≥ R_rigorous: approximation should be accurate")

    with col_plot:
        x = np.linspace(-1, 1, 500)
        # True functions
        cos_true = np.cos(M_val * x)
        sin_true = np.sin(M_val * x)

        # Jacobi-Anger truncated approximations
        cos_approx = np.zeros_like(x)
        sin_approx = np.zeros_like(x)
        bessel_vals = []
        for n in range(R_trunc + 1):
            Jn = bessel_j(n, M_val)
            bessel_vals.append(abs(Jn))
            coeffs_n = np.zeros(n + 1)
            coeffs_n[n] = 1.0
            Tn = chebval(x, coeffs_n)
            if n == 0:
                cos_approx += Jn * Tn
            elif n % 2 == 0:
                cos_approx += 2 * ((-1) ** (n // 2)) * Jn * Tn
            else:
                sin_approx += 2 * ((-1) ** ((n - 1) // 2)) * Jn * Tn

        if show_bessel:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

        for ax, true, approx, label in [
            (ax1, cos_true, cos_approx, f"cos({M_val:.1f}x)"),
            (ax2, sin_true, sin_approx, f"sin({M_val:.1f}x)"),
        ]:
            ax.plot(x, true, 'b-', lw=2, label='Exact', alpha=0.8)
            ax.plot(x, approx, 'r--', lw=1.8, label=f'J-A truncated (R={R_trunc})')
            ax.fill_between(x, true, approx, alpha=0.15, color='red')
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel("$x$"); ax.set_ylim(-1.3, 1.3)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
            ax.axhline(1, color='grey', lw=0.5, ls='--', alpha=0.4)
            ax.axhline(-1, color='grey', lw=0.5, ls='--', alpha=0.4)

        if show_bessel:
            ns = list(range(len(bessel_vals)))
            ax3.bar(ns, bessel_vals, color=['#4a90d9' if n <= R_trunc else '#ccc' for n in ns],
                    edgecolor='black', linewidth=0.5)
            ax3.axvline(R_trunc, color='red', lw=1.5, ls='--', label=f'R={R_trunc}')
            ax3.axvline(R_rigorous, color='green', lw=1.5, ls=':', label=f'R_rigorous={R_rigorous}')
            ax3.set_yscale('log')
            ax3.set_xlabel("$n$"); ax3.set_ylabel("$|J_n(M)|$")
            ax3.set_title("Bessel coefficient decay", fontsize=11, fontweight='bold')
            ax3.legend(fontsize=8); ax3.grid(True, alpha=0.2, which='both')

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.caption(
        "The Bessel coefficients |Jₙ(M)| decay rapidly for n > M (the 'cliff'). "
        "Truncating at R ≈ eM/2 gives exponentially small error. "
        "This is the analytic foundation behind the complexity bound in Helle et al. Lemma 5.2."
    )
    reference("Helle2025")

    key_concept(
        "Chebyshev polynomials are the <b>design language</b> for QSVT. "
        "They naturally satisfy the boundedness and parity constraints, converge rapidly, "
        "and their degree directly determines the quantum circuit depth. "
        "The Jacobi-Anger expansion provides <b>analytic Chebyshev coefficients</b> with "
        "rigorous degree bounds — no numerical fitting required."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017", "Low2019", "Helle2025"])
