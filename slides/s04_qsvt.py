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


def _rotate_2d(vec, angle):
    """Rotate a 2D vector by angle."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _evolve_state_2d(state, theta, alpha, apply_phase=False, phi=0.0):
    """Toy geometry for one query step.

    State is [x, y, z]:
    - (x, y) is the relevant signal plane
    - z is out-of-plane garbage

    Each U_A does two things:
    1) rotates in-plane by theta
    2) leaks amplitude out-of-plane by alpha

    Interleaved phases change the leakage axis and rotate previous garbage,
    allowing destructive interference in z.
    """
    x, y, z = state

    # In-plane action: every query rotates by theta in the relevant plane.
    xy_rot = _rotate_2d(np.array([x, y]), theta)

    # Tuned so interleaved path stays close to ideal, while naive drifts.
    leak_strength = 0.14 * np.sin(alpha)

    if apply_phase:
        # Phase steering rotates the leakage axis in-plane.
        leak_axis = np.array([np.cos(phi), np.sin(phi)])
        leak = leak_strength * np.dot(xy_rot, leak_axis)
        # Existing garbage is strongly damped under phase sequence, enabling cancellation.
        z_next = 0.22 * z + leak * np.cos(phi)
    else:
        # No phase steering: leakage keeps adding in one direction, so garbage compounds.
        leak = leak_strength * abs(xy_rot[0])
        z_next = 1.03 * z + leak

    # More out-of-plane weight means worse projected in-plane quality.
    # Interleaved path should remain accurate; naive path drifts more under z growth.
    proj_penalty = 0.03 if apply_phase else 0.25
    scale = max(0.05, 1.0 - proj_penalty * abs(z_next))
    xy_next = scale * xy_rot

    return np.array([xy_next[0], xy_next[1], z_next])


def _simulate_trajectories_2d(depth, sigma, phase_span):
    """Track naive vs interleaved trajectories in [x, y, z] geometry."""
    theta = 0.35
    alpha = np.arccos(np.clip(sigma, -1.0, 1.0))

    n_states = [np.array([1.0, 0.0, 0.0])]
    p_states = [np.array([1.0, 0.0, 0.0])]
    ideal_states = [np.array([1.0, 0.0, 0.0])]

    if depth <= 1:
        phases = np.array([phase_span])
    else:
        t = np.linspace(0.0, 1.0, depth)
        phases = phase_span * np.cos(np.pi * t)

    for idx in range(depth):
        n_states.append(_evolve_state_2d(n_states[-1], theta, alpha, apply_phase=False))
        p_states.append(_evolve_state_2d(p_states[-1], theta, alpha, apply_phase=True, phi=phases[idx]))

        # Ideal action: only in-plane rotation, no out-of-plane leakage.
        ideal_xy = _rotate_2d(ideal_states[-1][:2], theta)
        ideal_states.append(np.array([ideal_xy[0], ideal_xy[1], 0.0]))

    return np.array(n_states), np.array(p_states), np.array(ideal_states), phases, theta, alpha


def _draw_geometry_frame(depth, frame, sigma, phase_span):
    """Draw 3D trajectories with out-of-plane leakage and projected-state error."""
    n_states, p_states, ideal_states, phases, theta, alpha = _simulate_trajectories_2d(depth, sigma, phase_span)
    k = min(frame + 1, depth)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0], projection="3d"), fig.add_subplot(gs[0, 1], projection="3d")]

    z_all = np.concatenate([n_states[:k+1, 2], p_states[:k+1, 2]])
    z_max = max(0.25, float(np.max(np.abs(z_all))) * 1.15)

    for ax_idx, (ax, label, states, col_traj, col_arrow) in enumerate([
        (axes[0], "Naive: repeated $U_A$", n_states, "#e67e22", "#ff6b35"),
        (axes[1], "Interleaved: $D(\\phi_j)U_A$", p_states, "#2e7d32", "#1b5e20"),
    ]):
        ax.grid(True, alpha=0.25)
        ax.view_init(elev=28, azim=-55)

        xy = states[:k+1, :2]
        z_vals = states[:k+1, 2]
        z_abs = np.abs(z_vals)
        ideal_xy = ideal_states[:k+1, :2]

        # Draw the relevant signal plane z=0 and ideal in-plane unit-circle trajectory.
        plane_u = np.linspace(-1.05, 1.05, 20)
        plane_v = np.linspace(-1.05, 1.05, 20)
        uu, vv = np.meshgrid(plane_u, plane_v)
        ax.plot_surface(uu, vv, np.zeros_like(uu), color="#dbeafe", alpha=0.15, linewidth=0)

        t = np.linspace(0.0, 2.0 * np.pi, 160)
        ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), "--", color="#8aa1b1", linewidth=1.0, alpha=0.7)

        ax.plot(ideal_xy[:, 0], ideal_xy[:, 1], np.zeros(k + 1), "--", color="#4a90d9", linewidth=2.2,
                alpha=0.9, label="Ideal in-plane target")
        ax.plot(xy[:, 0], xy[:, 1], z_vals, "-", color=col_traj, linewidth=3.0, alpha=0.95,
                label="Actual 3D state")

        # Step markers are colored by out-of-plane leakage magnitude.
        z_norm = z_abs / (np.max(z_abs) + 1e-9)
        for i in range(k + 1):
            dot_color = plt.cm.YlOrRd(0.25 + 0.7 * z_norm[i])
            ax.plot([xy[i, 0]], [xy[i, 1]], [z_vals[i]], "o", color=dot_color, markersize=6,
                    alpha=0.95, markeredgecolor="black", markeredgewidth=0.4)
            ax.text(xy[i, 0] + 0.02, xy[i, 1] + 0.02, z_vals[i] + 0.01, str(i), fontsize=7, alpha=0.75)

            # Projection back to relevant plane.
            ax.plot([xy[i, 0], xy[i, 0]], [xy[i, 1], xy[i, 1]], [z_vals[i], 0.0],
                    color=col_arrow, linewidth=1.0, alpha=0.35)

            # Error from ideal projected point at same step.
            ax.plot([ideal_xy[i, 0], xy[i, 0]], [ideal_xy[i, 1], xy[i, 1]], [0.0, 0.0],
                    color="#64748b", linewidth=1.0, alpha=0.35)

        ax.plot([ideal_xy[k, 0]], [ideal_xy[k, 1]], [0.0], "*", color="#1f4db8", markersize=12,
                label="Ideal final (in-plane)", zorder=5)
        ax.plot([xy[k, 0]], [xy[k, 1]], [z_vals[k]], "X", color=col_traj, markersize=10,
                markeredgecolor="black", markeredgewidth=1.0, label="Actual final")
        ax.plot([xy[k, 0]], [xy[k, 1]], [0.0], "X", color=col_traj, markersize=9,
                markeredgecolor="black", markeredgewidth=0.8, label="Projected final")

        ax.set_xlabel("Plane axis 1", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("Plane axis 2", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_zlabel("Out-of-plane z", fontsize=10, fontweight="bold", labelpad=5)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=8, loc="upper left")

        ax.set_xlim(-1.10, 1.10)
        ax.set_ylim(-1.10, 1.10)
        ax.set_zlim(-0.10 * z_max, z_max)

        z_curr = abs(states[k, 2])
        err_curr = np.linalg.norm(states[k, :2] - ideal_states[k, :2])
        ax.text2D(0.98, 0.98,
              f"|z|={z_curr:.3f}\\nproj error={err_curr:.3f}",
              transform=ax.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#bbb"))

        if ax_idx == 1 and k > 0:
            ax.text2D(0.98, 0.02, f"$\\phi_j$ = {phases[k-1]:+.2f} rad",
                      transform=ax.transAxes, ha="right", va="bottom",
                      fontsize=10, bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", alpha=0.85, edgecolor="#bbb"))

    # Summary stats
    n_z_final = abs(n_states[k, 2])
    p_z_final = abs(p_states[k, 2])
    n_proj_err = np.linalg.norm(n_states[k, :2] - ideal_states[k, :2])
    p_proj_err = np.linalg.norm(p_states[k, :2] - ideal_states[k, :2])

    fig.text(0.5, 0.08,
            f"Step {k}/{depth} | "
            f"NAIVE: |z|={n_z_final:.3f}, proj_err={n_proj_err:.3f}  |  "
            f"INTERLEAVED: |z|={p_z_final:.3f}, proj_err={p_proj_err:.3f}  |  "
            f"Δ|z|={p_z_final-n_z_final:+.3f}, Δerr={p_proj_err-n_proj_err:+.3f}",
            ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.6",
            facecolor="#f0f0f0", alpha=0.9, edgecolor="#999"), family="monospace")

    fig.suptitle(
        f"QSVT Geometry in 3D: in-plane rotation θ={theta:.2f}, leakage angle α={alpha:.2f}",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.15, top=0.90, wspace=0.20)
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
        "True 3D state-space view: (x, y) is the relevant signal plane, z is out-of-plane garbage. "
        "Each $U_A$ rotates by $\\theta$ in-plane and leaks by $\\alpha$ out-of-plane; post-selection projects back to z=0."
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

    st.info(
        "💡 **The QSVT insight:** Every query rotates the useful state in-plane but also leaks amplitude out-of-plane. "
        "Without phase steering, that leakage compounds and projection back to the plane gives the wrong vector. "
        "With interleaved phases, out-of-plane leakage destructively interferes, so projection recovers a much more accurate in-plane state."
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
        r"Here $p$ is initial success probability, $\kappa$ is condition number, and $\epsilon$ is approximation error."
    )

    key_concept(
        "QSVT transforms each singular value $\\sigma_i \\mapsto P(\\sigma_i)$ by interleaving "
        "a block encoding with phase rotations. The phases create <b>destructive interference</b> "
        "that eliminates garbage cross-terms — giving an exact polynomial transformation in "
        "$O(d)$ queries. Different polynomials $P$ yield different quantum algorithms."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017"])
