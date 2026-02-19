"""Slide 4: QSVT — How it fixes the block-encoding problem."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from slides.components import (
    slide_header, key_concept, reference_list,
    qsvt_circuit_schematic,
)

TITLE = "QSVT: The Grand Unification"


# ── Visual: QSVT signal processing intuition ─────────────────────────────────

def _draw_qsvt_intuition():
    """Side-by-side comparison: naive repeated application vs QSVT interleaved phases."""

    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # ── Left panel: naive repeated application ──
    ax_naive = fig.add_subplot(gs[0, 0])
    ax_naive.set_xlim(-0.5, 7.5)
    ax_naive.set_ylim(-1.0, 2.2)
    ax_naive.set_aspect('equal')
    ax_naive.axis('off')
    ax_naive.set_title("Naive: repeated $U_A$", fontsize=12, fontweight='bold',
                       color='#c62828', pad=10)

    # Wires
    ax_naive.plot([-0.3, 7.3], [1.5, 1.5], 'k-', lw=1.2)
    ax_naive.plot([-0.3, 7.3], [0.0, 0.0], 'k-', lw=1.2)
    ax_naive.text(-0.5, 1.5, "anc", ha='right', va='center', fontsize=9, style='italic')
    ax_naive.text(-0.5, 0.0, "sys", ha='right', va='center', fontsize=9, style='italic')

    for i, x0 in enumerate([1.0, 3.5, 6.0]):
        rect = patches.FancyBboxPatch((x0 - 0.4, -0.35), 0.8, 2.2,
                                       boxstyle="round,pad=0.08",
                                       facecolor='#4a90d9', alpha=0.85,
                                       edgecolor='black', linewidth=1.2)
        ax_naive.add_patch(rect)
        ax_naive.text(x0, 0.75, r"$U_A$", ha='center', va='center',
                      fontsize=11, color='white', fontweight='bold')

    # Garbage arrows leaking
    for x0 in [1.7, 4.2]:
        ax_naive.annotate("", xy=(x0 + 0.8, 1.5), xytext=(x0, 0.0),
                          arrowprops=dict(arrowstyle='->', color='#ff9800',
                                          lw=1.8, linestyle='--'))
    ax_naive.text(3.5, -0.8, "garbage leaks back each step",
                  ha='center', fontsize=9, color='#c62828', fontweight='bold')

    # ── Right panel: QSVT interleaved phases ──
    ax_qsvt = fig.add_subplot(gs[0, 1])
    ax_qsvt.set_xlim(-0.5, 10.5)
    ax_qsvt.set_ylim(-1.0, 2.2)
    ax_qsvt.set_aspect('equal')
    ax_qsvt.axis('off')
    ax_qsvt.set_title("QSVT: interleaved phase rotations", fontsize=12,
                       fontweight='bold', color='#2e7d32', pad=10)

    # Wires
    ax_qsvt.plot([-0.3, 10.3], [1.5, 1.5], 'k-', lw=1.2)
    ax_qsvt.plot([-0.3, 10.3], [0.0, 0.0], 'k-', lw=1.2)
    ax_qsvt.text(-0.5, 1.5, "anc", ha='right', va='center', fontsize=9, style='italic')
    ax_qsvt.text(-0.5, 0.0, "sys", ha='right', va='center', fontsize=9, style='italic')

    x = 0.5
    for i in range(3):
        # Phase rotation on ancilla
        phi_rect = patches.FancyBboxPatch((x - 0.35, 1.2), 0.7, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor='#7b61ff', alpha=0.9,
                                           edgecolor='black', linewidth=1.0)
        ax_qsvt.add_patch(phi_rect)
        ax_qsvt.text(x, 1.5, f"$\\phi_{i}$", ha='center', va='center',
                      fontsize=9, color='white', fontweight='bold')
        x += 1.5

        # U_A block
        ua_rect = patches.FancyBboxPatch((x - 0.4, -0.35), 0.8, 2.2,
                                          boxstyle="round,pad=0.08",
                                          facecolor='#4a90d9', alpha=0.85,
                                          edgecolor='black', linewidth=1.2)
        ax_qsvt.add_patch(ua_rect)
        label = r"$U_A$" if i % 2 == 0 else r"$U_A^\dagger$"
        ax_qsvt.text(x, 0.75, label, ha='center', va='center',
                      fontsize=10, color='white', fontweight='bold')
        x += 1.5

    # Final phase
    phi_rect = patches.FancyBboxPatch((x - 0.35, 1.2), 0.7, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#7b61ff', alpha=0.9,
                                       edgecolor='black', linewidth=1.0)
    ax_qsvt.add_patch(phi_rect)
    ax_qsvt.text(x, 1.5, f"$\\phi_3$", ha='center', va='center',
                  fontsize=9, color='white', fontweight='bold')

    # Shield annotation
    ax_qsvt.text(5.0, -0.8, "phases project away garbage at every step",
                 ha='center', fontsize=9, color='#2e7d32', fontweight='bold')

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render():
    slide_header("QSVT: The Grand Unification",
                 "One framework to rule (almost) all quantum algorithms")

    # ── Recap: the problem ──
    st.markdown(r"""
### Recall: the block-encoding problem

We showed that naively multiplying $U_A \cdot U_A$ lets garbage cross-terms ($B \cdot C$) leak
back into the signal subspace.  Every additional application makes this contamination **worse**.

**QSVT's key insight:** instead of multiplying $U_A$ blindly, **interleave** each application  
with a phase rotation $e^{i\phi_j Z}$ on the ancilla qubit.  These phases act as a "filter" that  
keeps the computation in the clean signal subspace throughout the entire circuit.
""")

    # ── Visual comparison ──
    _draw_qsvt_intuition()

    st.markdown("---")

    # ── How it works ──
    st.markdown(r"""
### How the phase rotations help

Consider the singular value decomposition $A/\alpha = \sum_i \sigma_i |u_i\rangle\langle v_i|$.

After each $U_A$, the ancilla partially rotates between $|0\rangle$ (signal) and $|\perp\rangle$ (garbage)
by an angle that depends on $\sigma_i$.  The phase gates $e^{i\phi_j Z}$ apply **different phases**
to the $|0\rangle$ and $|\perp\rangle$ components, which controls the interference pattern.

By choosing the $d+1$ angles $\{\phi_0, \dots, \phi_d\}$ correctly, the net effect after $d$
applications of $U_A$ is:

$$\sigma_i \;\longmapsto\; P(\sigma_i)$$

where $P$ is a **degree-$d$ polynomial** that we control.  The garbage terms
destructively interfere and cancel exactly — no contamination.
""")

    st.markdown("---")

    # ── The central theorem ──
    col_thm, col_circ = st.columns([1.1, 1])

    with col_thm:
        st.markdown(r"""
### The Central Theorem

Given an $(\alpha, a)$-block encoding $U_A$ of $A$, and a polynomial $P$ of degree $d$ satisfying:

1. $|P(x)| \leq 1$ for all $x \in [-1, 1]$
2. $P$ has **definite parity** (even or odd)

there exist angles $\{\phi_0, \phi_1, \dots, \phi_d\}$ such that:

$$
(\langle 0| \otimes I)\;\left[\prod_{j} e^{i\phi_j Z}\, U_A^{(\dagger)}\right]\;(|0\rangle \otimes I) = P(A/\alpha)
$$

The result is an **exact** block encoding of $P(A/\alpha)$ — not an approximation to the circuit, but
exact realisation of whatever polynomial we choose.
""")

    with col_circ:
        st.markdown("#### QSVT circuit (degree 6)")
        fig_circ = qsvt_circuit_schematic(6, figsize=(12, 2.5))
        st.pyplot(fig_circ, use_container_width=True)
        plt.close(fig_circ)

        st.markdown(r"""
        Query complexity: **exactly $d$** applications of $U_A$.  
        The polynomial $P$ determines the algorithm;  
        the angles $\{\phi_j\}$ are found by **classical preprocessing**.
        """)

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
        "QSVT transforms each singular value $\\sigma_i \\mapsto P(\\sigma_i)$ by interleaving "
        "a block encoding with phase rotations. The phases create <b>destructive interference</b> "
        "that eliminates garbage cross-terms — giving an exact polynomial transformation in "
        "$O(d)$ queries. Different polynomials $P$ yield different quantum algorithms."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017"])
