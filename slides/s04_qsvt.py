"""Slide 4: QSVT — How it fixes the block-encoding problem."""

import time
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


def _evolve_state_2d(state, sigma, beta, apply_phase=False, phi=0.0):
    """Apply U_A (optionally with phase gate) to [signal, garbage] state.
    
    Key insight: phase gates create OPPOSITE rotations on signal vs garbage.
    - Signal couples to |0⟩: rotation by e^{i*phi}
    - Garbage couples to |1⟩: rotation by e^{-i*phi}
    
    This opposite rotation creates destructive interference for garbage while
    keeping signal constructive.
    """
    s_val = state[0]
    g_prev = state[1]  # Full complex garbage state
    
    # U_A: leak signal magnitude to garbage
    leak_magnitude = beta * np.real(s_val)
    
    # Phase gate creates opposite rotations:
    # Signal: +phi, Garbage: -phi for destructive interference
    if apply_phase:
        phase_rotation = np.exp(1j * phi)
        antirotation = np.exp(-1j * phi)
    else:
        phase_rotation = 1.0
        antirotation = 1.0
    
    # Signal decays with feedback from garbage, then rotates
    s_next = (sigma * np.real(s_val) - 0.1 * np.real(g_prev)) * phase_rotation
    
    # Garbage accumulates leakage and receives opposite phase for destructive interference
    g_next = g_prev * antirotation + leak_magnitude * antirotation
    
    return np.array([s_next, g_next])


def _simulate_trajectories_2d(depth, sigma, phase_span):
    """Track full state trajectory for both methods.
    
    Key difference:
    - Naive: garbage always leaks in direction 0 → accumulates upward
    - Interleaved: garbage leaks in directions that vary → oscillations
    """
    beta = np.sqrt(max(0.0, 1.0 - sigma**2))
    
    n_states = [np.array([1.0 + 0.0j, 0.0 + 0.0j])]
    p_states = [np.array([1.0 + 0.0j, 0.0 + 0.0j])]
    
    # Phase schedule that oscillates
    if depth <= 1:
        phases = np.array([phase_span])
    else:
        t = np.linspace(0.0, 1.0, depth)
        phases = phase_span * np.cos(np.pi * t)
    
    for idx in range(depth):
        # Naive: phase is always 0, garbage leaks in same direction
        n_next = _evolve_state_2d(n_states[-1], sigma, beta, apply_phase=False)
        n_states.append(n_next)
        
        # Interleaved: phase varies, garbage leaks in different directions
        p_next = _evolve_state_2d(p_states[-1], sigma, beta, apply_phase=True, phi=phases[idx])
        p_states.append(p_next)
    
    return np.array(n_states), np.array(p_states), phases


def _draw_geometry_frame(depth, frame, sigma, phase_span):
    """Draw 2D signal-garbage geometry showing accumulation with and without phase steering."""
    n_states, p_states, phases = _simulate_trajectories_2d(depth, sigma, phase_span)
    k = min(frame + 1, depth)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.3})
    
    for ax_idx, (ax, label, states, col_traj, col_arrow) in enumerate([
        (axes[0], "Naive: repeated $U_A$", n_states, "#e67e22", "#ff6b35"),
        (axes[1], "Interleaved: $D(\\phi_j)U_A$", p_states, "#2e7d32", "#1b5e20"),
    ]):
        # Draw axis labels and grid
        ax.axhline(0, color="#999", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#999", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")
        
        # Extract real and imaginary parts of garbage for 2D plot
        # Signal is on x-axis, garbage oscillates in complex plane
        s_vals = np.real(states[:k+1, 0])
        g_real = np.real(states[:k+1, 1])
        g_imag = np.imag(states[:k+1, 1])
        g_mag = np.sqrt(g_real**2 + g_imag**2)
        
        # Plot trajectory in (signal, garbage-magnitude) space
        ax.plot(s_vals, g_mag, "-", color=col_traj, linewidth=3, alpha=0.8, 
               label="Full state evolution", zorder=2)
        
        # Draw arrows at each step
        for i in range(k):
            if i < len(states) - 1:
                x0, y0 = s_vals[i], g_mag[i]
                x1, y1 = s_vals[i+1], g_mag[i+1]
                dx, dy = x1 - x0, y1 - y0
                if np.sqrt(dx**2 + dy**2) > 0.005:
                    ax.arrow(x0, y0, dx*0.85, dy*0.85, head_width=0.03, head_length=0.04,
                            fc=col_arrow, ec=col_arrow, alpha=0.9, zorder=3, linewidth=1.2)
        
        # Plot dots at each step
        for i in range(k+1):
            s_mag = abs(s_vals[i])
            total = s_mag + g_mag[i] + 0.001
            ratio = s_mag / total
            dot_color = plt.cm.RdYlBu(0.7 + 0.3*ratio)  
            ax.plot(s_vals[i], g_mag[i], 'o', color=dot_color, markersize=7, 
                   alpha=0.85, zorder=4, markeredgecolor="black", markeredgewidth=0.6)
            ax.text(s_vals[i] + 0.02, g_mag[i] + 0.02, str(i), fontsize=7, alpha=0.6)
        
        # For interleaved, show garbage direction field
        if ax_idx == 1:
            for i in range(k):
                if g_mag[i] > 0.01:
                    ax.arrow(s_vals[i], 0.01, g_real[i]*0.15, g_imag[i]*0.15,
                            head_width=0.015, head_length=0.015,
                            fc="#666", ec="#999", alpha=0.25, zorder=1, linewidth=0.8)
        
        # Post-selection projection
        current_s = s_vals[k]
        current_g = g_mag[k]
        ax.plot([current_s, current_s], [current_g, 0], "--", color="#4a90d9", 
               linewidth=2, alpha=0.6, label="Measure signal", zorder=1)
        ax.plot(current_s, 0, "X", color="#4a90d9", markersize=12, alpha=0.85, 
               markeredgewidth=2, markeredgecolor="navy", zorder=5)
        
        # Labels and limits
        ax.set_xlabel("Signal amplitude (real)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Garbage magnitude |G|", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="upper left")
        
        lim = max(0.2, max(np.max(np.abs(s_vals)), np.max(g_mag)) * 1.3)
        ax.set_xlim(-0.05, lim)
        ax.set_ylim(-0.05, lim)
        
        if ax_idx == 1 and k > 0:
            ax.text(0.98, 0.02, f"$\\phi_j$ = {phases[k-1]:+.2f} rad",
                   transform=ax.transAxes, ha="right", va="bottom",
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                   facecolor="white", alpha=0.85, edgecolor="#bbb"))
    
    # Summary stats
    n_s_final = abs(n_states[k, 0])
    n_g_final = abs(n_states[k, 1])
    p_s_final = abs(p_states[k, 0])
    p_g_final = abs(p_states[k, 1])
    
    fig.text(0.5, 0.08,
            f"Step {k}/{depth} | "
            f"NAIVE: signal={n_s_final:.3f}, |garbage|={n_g_final:.3f}  |  "
            f"INTERLEAVED: signal={p_s_final:.3f}, |garbage|={p_g_final:.3f}  |  "
            f"Δsignal={p_s_final-n_s_final:+.3f}, Δ|garbage|={p_g_final-n_g_final:+.3f}",
            ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.6",
            facecolor="#f0f0f0", alpha=0.9, edgecolor="#999"), family="monospace")
    
    fig.suptitle(
        f"QSVT: Garbage Accumulation Without vs With Phase Steering (σ={sigma:.2f})",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    return fig


def render():
    slide_header("QSVT: The Grand Unification",
                 "One framework to rule (almost) all quantum algorithms")

    # ── Recap: the problem ──
    st.markdown(r"""
### Recall: the block-encoding problem

Naively multiplying $U_A$ causes garbage cross-terms ($B\cdot C$) to leak
back into the signal block, and leakage compounds with depth.

**QSVT insight:** instead of blindly repeating $U_A$, interleave each query
with a phase rotation $e^{i\phi_j Z}$ on the ancilla qubit.
The phases steer interference: keep signal, cancel garbage.
""")

    # ── Visual comparison ──
    _draw_qsvt_intuition()

    # ── How it works ──
    st.markdown(r"""
### How the phase rotations help

Consider the singular value decomposition $A/\alpha = \sum_i \sigma_i |u_i\rangle\langle v_i|$.

Each $U_A$ mixes ancilla signal $|0\rangle$ and garbage $|\perp\rangle$ by an amount that
depends on $\sigma_i$. Phase gates then shift these branches differently.

Concretely, on ancilla states:

$$
e^{i\phi_j Z}|0\rangle = e^{i\phi_j}|0\rangle,\qquad
e^{i\phi_j Z}|\perp\rangle = e^{-i\phi_j}|\perp\rangle
$$

So one angle sequence can make signal branches constructive and garbage branches destructive.

By choosing the $d+1$ angles $\{\phi_0, \dots, \phi_d\}$ correctly, the net effect after $d$
applications of $U_A$ is:

$$\sigma_i \;\longmapsto\; P(\sigma_i)$$

where $P$ is a **degree-$d$ polynomial** that we control.  The garbage terms
destructively interfere and cancel exactly — no contamination.
""")

    st.caption(
        "Mental model: QSVT is phase-engineered interference on ancilla-conditioned paths. "
        "The polynomial is the resulting transfer function on singular values."
    )

    st.markdown("---")
    st.markdown("### Animated Intuition: Post-Selection & Phase Steering")
    st.caption(
        "2D state-space visualization: X-axis is signal amplitude, Y-axis is garbage amplitude. "
        "See how U_A mixes them, and how phase gates keep them separated."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        toy_depth = st.slider("Depth (# of U_A applications)", 2, 16, 8, 1, key="s04_toy_depth")
    with c2:
        toy_sigma = st.slider("Singular value σ", 0.55, 0.95, 0.75, 0.02, key="s04_toy_sigma")
    with c3:
        toy_span = st.slider("Phase span", 0.20, 1.40, 0.80, 0.05, key="s04_toy_span")

    f_col, p_col = st.columns([2, 1])
    with f_col:
        frame = st.slider("Current step", 0, toy_depth - 1, toy_depth - 1, 1, key="s04_toy_frame")
    with p_col:
        play = st.button("Play ▶", key="s04_toy_play", use_container_width=True)

    ph = st.empty()
    if play:
        for fr in range(toy_depth):
            fig_geom = _draw_geometry_frame(toy_depth, fr, toy_sigma, toy_span)
            ph.pyplot(fig_geom, use_container_width=True)
            plt.close(fig_geom)
            time.sleep(0.12)
    else:
        fig_geom = _draw_geometry_frame(toy_depth, frame, toy_sigma, toy_span)
        ph.pyplot(fig_geom, use_container_width=True)
        plt.close(fig_geom)

    with st.expander("📊 How to read this visualization", expanded=True):
        st.markdown("""
**Key idea:** Garbage doesn't just *grow*—its *direction* matters for interference.

**Left panel (Naive):** Repeated $U_A$ only
- Orange trajectory climbs monotonically (garbage always leaks in the SAME direction).
- Colored dots: state after each application (red = garbage-rich, blue = signal-rich).
- Blue X: where the signal is measured after post-selection.
- **Problem:** Since garbage always leaks in the same direction, it compounds: magnitudes keep growing.

**Right panel (Interleaved):** $U_A$ with varying phase gates $D(\\phi_j)$
- Phase $\\phi_j$ rotates the direction that garbage leaks to.
- Green trajectory shows this: garbage leaks in *different* directions each step.
- The small gray arrows (in complex plane) show the direction of garbage at each step—they rotate!
- **Benefit:** When garbage leaks in different directions, the real and imaginary parts can oscillate and partially cancel.
- Result: |garbage| stays much smaller than naive case.

**Bottom summary:** Shows $\\Delta|\\text{garbage}|$ = how much smaller the interleaved garbage is.  
When this is negative, phase steering is *suppressing* garbage compared to naive.
        """)

    st.info(
        "💡 **The QSVT insight:** Phase gates rotate garbage so it doesn't accumulate monotonically. "
        "Without them, each $U_A$ pushes garbage further in the same direction. "
        "With phase steering, garbage oscillates—the real and imaginary parts fight each other, reducing net magnitude."
    )


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

    c_left, c_right = st.columns([1.25, 1])
    with c_left:
        st.success(
            "**Angle-Design Checklist**\n"
            "1. Pick target transform on singular values\n"
            "2. Approximate with bounded polynomial $P$ on [-1, 1]\n"
            "3. Enforce parity (even/odd or split into both)\n"
            "4. Solve for phase angles $\\{\\phi_j\\}$ classically\n"
            "5. Build phase-interleaved circuit and validate error"
        )
    with c_right:
        st.markdown(
            """
**Mental model**

- Polynomial design sets the transfer function
- Angle synthesis compiles that transfer function
- Circuit execution realizes it in $O(d)$ block-encoding queries
"""
        )

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

    st.caption(
        "Here $p$ is initial success probability, $\kappa$ is condition number, and $\epsilon$ is approximation error."
    )

    key_concept(
        "QSVT transforms each singular value $\\sigma_i \\mapsto P(\\sigma_i)$ by interleaving "
        "a block encoding with phase rotations. The phases create <b>destructive interference</b> "
        "that eliminates garbage cross-terms — giving an exact polynomial transformation in "
        "$O(d)$ queries. Different polynomials $P$ yield different quantum algorithms."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017"])
