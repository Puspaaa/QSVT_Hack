"""Slide 3: Block Encoding & LCU — the interface between classical matrices and quantum circuits."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from slides.components import (
    slide_header, reference, key_concept, matrix_heatmap, reference_list,
)

TITLE = "Block Encoding & LCU"


def _draw_mixing_diagram():
    """Show U_A x U_A = product, with block multiplication revealing contamination."""

    fig = plt.figure(figsize=(12, 4.2))

    # Layout: [U_A] [x] [U_A] [=] [Product]
    # Use gridspec for precise control
    gs = fig.add_gridspec(1, 7, width_ratios=[3, 0.5, 3, 0.5, 3, 0.1, 0.1],
                          wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])   # first U_A
    ax_x = fig.add_subplot(gs[0, 1])  # multiplication sign
    ax2 = fig.add_subplot(gs[0, 2])   # second U_A
    ax_eq = fig.add_subplot(gs[0, 3]) # equals sign
    ax3 = fig.add_subplot(gs[0, 4])   # product

    def _draw_block(ax, blocks, title, col_labels=None, row_labels=None):
        """Draw a 2x2 block matrix on the given axes."""
        sz = 4
        bg = plt.Rectangle((0, 0), sz, sz, facecolor='#f5f5f5',
                            edgecolor='black', linewidth=2)
        ax.add_patch(bg)
        # Grid line
        ax.plot([sz/2, sz/2], [0, sz], color='black', lw=0.8, alpha=0.5)
        ax.plot([0, sz], [sz/2, sz/2], color='black', lw=0.8, alpha=0.5)

        for (r, c, fc, alpha, lbl, tc) in blocks:
            x0 = c * sz/2
            y0 = (1 - r) * sz/2   # row 0 = top → y = sz/2
            rect = plt.Rectangle((x0, y0), sz/2, sz/2, facecolor=fc,
                                  alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
            ax.text(x0 + sz/4, y0 + sz/4, lbl, fontsize=11, ha='center',
                    va='center', color=tc, fontweight='bold')

        if col_labels:
            for i, cl in enumerate(col_labels):
                ax.text(sz/4 + i*sz/2, sz + 0.25, cl, fontsize=8,
                        ha='center', va='bottom', color=col_labels[cl])
        if row_labels:
            for i, rl in enumerate(row_labels):
                ax.text(-0.2, sz - sz/4 - i*sz/2, rl, fontsize=8,
                        ha='right', va='center', color=row_labels[rl])

        ax.set_xlim(-0.8, sz + 0.3)
        ax.set_ylim(-0.5, sz + 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    # ── First U_A ──
    _draw_block(ax1, [
        (0, 0, '#4a90d9', 0.85, r"$A/\alpha$", 'white'),
        (0, 1, '#90a4ae', 0.35, r"$B$", '#333'),
        (1, 0, '#90a4ae', 0.35, r"$C$", '#333'),
        (1, 1, '#90a4ae', 0.35, r"$D$", '#333'),
    ], r"$U_A$",
       {"$|0\\rangle$": '#4a90d9', "$|\\perp\\rangle$": '#999'},
       {"$\\langle 0|$": '#4a90d9', "$\\langle\\perp|$": '#999'})

    # ── Multiplication sign ──
    ax_x.set_xlim(0, 1); ax_x.set_ylim(0, 1); ax_x.axis('off')
    ax_x.text(0.5, 0.5, r"$\times$", fontsize=24, ha='center', va='center',
              color='#333', fontweight='bold')

    # ── Second U_A ──
    _draw_block(ax2, [
        (0, 0, '#4a90d9', 0.85, r"$A/\alpha$", 'white'),
        (0, 1, '#90a4ae', 0.35, r"$B$", '#333'),
        (1, 0, '#90a4ae', 0.35, r"$C$", '#333'),
        (1, 1, '#90a4ae', 0.35, r"$D$", '#333'),
    ], r"$U_A$")

    # ── Equals sign ──
    ax_eq.set_xlim(0, 1); ax_eq.set_ylim(0, 1); ax_eq.axis('off')
    ax_eq.text(0.5, 0.5, r"$=$", fontsize=24, ha='center', va='center',
               color='#333', fontweight='bold')

    # ── Product: top-left is contaminated ──
    sz = 4
    bg = plt.Rectangle((0, 0), sz, sz, facecolor='#f5f5f5',
                        edgecolor='black', linewidth=2)
    ax3.add_patch(bg)
    ax3.plot([sz/2, sz/2], [0, sz], color='black', lw=0.8, alpha=0.5)
    ax3.plot([0, sz], [sz/2, sz/2], color='black', lw=0.8, alpha=0.5)

    # Top-left block: split into desired + cross-term
    # Blue desired part (bottom of top-left cell)
    desired = plt.Rectangle((0, sz/2), sz/2, sz/4, facecolor='#4a90d9',
                              alpha=0.8, edgecolor='none')
    ax3.add_patch(desired)
    ax3.text(sz/4, sz/2 + sz/8, r"$(A/\alpha)^2$", fontsize=9.5,
             ha='center', va='center', color='white', fontweight='bold')

    # Orange cross-term (top of top-left cell)
    cross = plt.Rectangle((0, sz*3/4), sz/2, sz/4, facecolor='#ff9800',
                            alpha=0.8, edgecolor='none')
    ax3.add_patch(cross)
    ax3.text(sz/4, sz*3/4 + sz/8, r"$+ \; B \cdot C$", fontsize=9.5,
             ha='center', va='center', color='white', fontweight='bold')

    # Red contamination border around entire top-left
    contam_border = plt.Rectangle((0, sz/2), sz/2, sz/2, facecolor='none',
                                   edgecolor='#c62828', linewidth=3, linestyle='--')
    ax3.add_patch(contam_border)

    # Other blocks — grey with dots
    for (r, c) in [(0, 1), (1, 0), (1, 1)]:
        x0 = c * sz/2
        y0 = (1 - r) * sz/2
        rect = plt.Rectangle((x0, y0), sz/2, sz/2, facecolor='#90a4ae',
                              alpha=0.2, edgecolor='none')
        ax3.add_patch(rect)
        ax3.text(x0 + sz/4, y0 + sz/4, r"$\cdot$", fontsize=14,
                 ha='center', va='center', color='#666')

    ax3.set_xlim(-0.8, sz + 0.3)
    ax3.set_ylim(-0.5, sz + 0.6)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title(r"$U_A^2$  (top-left block)", fontsize=11, fontweight='bold', pad=8)

    # Annotation arrows
    ax3.annotate("desired", xy=(sz/4, sz/2 + sz/8), xytext=(sz/4, -0.3),
                 fontsize=8, color='#1565c0', ha='center', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.2))
    ax3.annotate("garbage\nleak-back", xy=(sz/4, sz*3/4 + sz/8),
                 xytext=(sz/2 + 0.8, sz + 0.35),
                 fontsize=8, color='#e65100', ha='center', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.2))

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        r"Block multiplication: the top-left entry of $U_A^2$ is "
        r"$(A/\alpha)^2 + B\cdot C$. The cross-term $B\cdot C$ (orange) is "
        r"garbage that couples back into the signal subspace — this is why "
        r"naive repeated application gives the wrong answer."
    )


def render():
    slide_header("Block Encoding & LCU",
                 "Embedding non-unitary matrices inside quantum circuits")

    # ── Motivation ──
    st.markdown(r"""
### The Problem: non-unitary operations on a quantum computer

In physics we frequently need to apply a matrix $A$ (e.g.\ a Hamiltonian time-step,
a finite-difference stencil, a Green's function) to a state.
But quantum circuits can only apply **unitary** operators — they must preserve norm.

If $A$ is not unitary (e.g.\ $\|A\| < 1$, or $A$ is not even square-normal),
we **cannot** implement it directly as a gate.  
**Block encoding** solves this by hiding $A$ inside a larger unitary.
""")

    # ── Definition + Visual side by side ──
    col_def, col_vis = st.columns([1.2, 1])

    with col_def:
        st.markdown(r"""
### Block Encoding Definition

A unitary $U_A$ on $a + n$ qubits is an **$(\alpha,\, a)$-block encoding** of $A$ if

$$
A = \alpha\; (\langle 0|^{\otimes a} \otimes I)\; U_A\; (|0\rangle^{\otimes a} \otimes I)
$$

The normalisation factor $\alpha \geq \|A\|$ ensures the block $A/\alpha$ has
operator norm $\leq 1$ so it can fit inside a unitary.

**Recipe:**
1. Prepare ancilla in $|0\rangle^{\otimes a}$
2. Apply $U_A$ to ancilla + data register
3. **Postselect** ancilla on $|0\rangle^{\otimes a}$

On success (probability $\|A|\psi\rangle\|^2 / \alpha^2$) the data register contains $A|\psi\rangle / \alpha$.
""")

    with col_vis:
        st.markdown("#### Block structure of $U_A$")

        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        rect_full = plt.Rectangle((0, 0), 4, 4, facecolor='#e0e0e0',
                                   edgecolor='black', linewidth=2)
        ax.add_patch(rect_full)
        rect_a = plt.Rectangle((0, 2), 2, 2, facecolor='#4a90d9', alpha=0.85,
                                edgecolor='black', linewidth=2)
        ax.add_patch(rect_a)
        ax.text(1, 3, r"$A/\alpha$", fontsize=18, ha='center', va='center',
                color='white', fontweight='bold')
        ax.text(3, 3, r"$\cdot$", fontsize=18, ha='center', va='center', color='#666')
        ax.text(1, 1, r"$\cdot$", fontsize=18, ha='center', va='center', color='#666')
        ax.text(3, 1, r"$\cdot$", fontsize=18, ha='center', va='center', color='#666')
        ax.text(1, 4.3, "ancilla = |0>", fontsize=10, ha='center', color='#4a90d9')
        ax.text(3, 4.3, "ancilla != |0>", fontsize=10, ha='center', color='#999')
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.8)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Unitary $U_A$", fontsize=13, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # ── The repeated-application problem ──
    st.markdown(r"""
### Why a single block encoding is not enough

Suppose we want to apply a **polynomial** of $A$, e.g.\ $p(A) = A^k$ (time evolution).  
A naive approach: just apply $U_A$ repeatedly $k$ times with postselection after each step.

**The problem — data/garbage mixing:**

$$
U_A\;|0\rangle|{\psi}\rangle \;=\; |0\rangle\,\frac{A}{\alpha}|\psi\rangle
\;\;+\;\; |{\perp}\rangle\,|\text{garbage}\rangle
$$

After the first application the state has a **garbage component** in the ancilla-$|\perp\rangle$ subspace.
Applying $U_A$ a second time mixes the clean signal with the garbage — the ancilla $|0\rangle$ subspace
now contains contributions from **both** the desired $A^2|\psi\rangle$ piece **and** cross terms that
couple back from the garbage.

Each subsequent application makes the contamination worse.  Postselecting at the end gives
the wrong answer, and the success probability falls **exponentially** as $(\|A\|/\alpha)^{2k}$.
""")

    # ── Visual: one application vs two ────────────────────────────────────
    _draw_mixing_diagram()

    st.info(
        "**This is the core motivation for QSVT:**  "
        "QSVT provides a way to apply an *arbitrary polynomial* $p(A)$ using "
        "the block encoding, **without** ever letting the garbage leak back "
        "into the signal subspace. It does so by interleaving $U_A$ with "
        "carefully chosen phase rotations on the ancilla."
    )

    st.markdown("---")

    # ── LCU ──
    st.markdown(r"""
### How to build a block encoding: Linear Combination of Unitaries (LCU)

Decompose $A = \sum_{i} \alpha_i\, U_i$ where each $U_i$ is easy to implement.  Then:

1. **PREPARE** $|0\rangle \mapsto \sum_i \sqrt{\alpha_i / s}\; |i\rangle$, with $s = \sum |\alpha_i|$  
2. **SELECT** $|i\rangle|\psi\rangle \mapsto |i\rangle\, U_i|\psi\rangle$  
3. $U_A = \text{PREPARE}^\dagger \cdot \text{SELECT} \cdot \text{PREPARE}$ is an $(s, a)$-block encoding of $A$

For our diffusion operator: $A_{\text{diff}} = a_0\, I + a_+\, S + a_-\, S^\dagger$ with just **2 ancilla qubits**.
""")

    key_concept(
        "Block encoding is the <b>interface</b> between classical linear algebra and quantum circuits. "
        "But applying the block encoding repeatedly mixes signal and garbage -- "
        "this motivates <b>QSVT</b>, which applies matrix polynomials cleanly via phase-interleaved circuits."
    )

    reference_list(["Gilyen2019", "Berry2015"])
