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

# Try importing QSP angle solver for the exact model tab.
try:
    from solvers import cvx_poly_coef, Angles_Fixed
    _HAS_QSP = True
except Exception:
    _HAS_QSP = False

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


# ── True 2×2 QSP model functions ─────────────────────────────────────────────

def _qsp_block_encoding(sigma):
    """Return the 2×2 block encoding W(σ) for a single singular value."""
    s = np.sqrt(max(0.0, 1.0 - sigma ** 2))
    return np.array([[sigma, s], [-s, sigma]], dtype=complex)


def _phase_gate(phi):
    """Return the 2×2 phase gate D(φ) = diag(e^{iφ}, e^{-iφ})."""
    return np.diag([np.exp(1j * phi), np.exp(-1j * phi)])


def _simulate_qsp(sigma, phases, use_phases=True):
    """Run the QSP sequence step-by-step on initial state |0⟩ = [1, 0].

    Returns arrays of shape (d+1, 2): signal_prob and garbage_prob at each step,
    plus the final P(σ) value.

    The QSP convention: D(φ₀) · W · D(φ₁) · W† · D(φ₂) · W · ...
    alternating W and W† after each phase gate.
    """
    W = _qsp_block_encoding(sigma)
    W_dag = W.conj().T
    state = np.array([1.0, 0.0], dtype=complex)  # |0⟩

    signal_probs = [1.0]
    garbage_probs = [0.0]

    d = len(phases)
    for j in range(d):
        # Apply phase gate
        phi = phases[j] if use_phases else 0.0
        state = _phase_gate(phi) @ state
        # Apply W or W† (alternating)
        if j % 2 == 0:
            state = W @ state
        else:
            state = W_dag @ state
        signal_probs.append(float(abs(state[0]) ** 2))
        garbage_probs.append(float(abs(state[1]) ** 2))

    # Final P(σ) is the (0,0) element of the full product, i.e. state[0] if we
    # started from |0⟩.
    p_sigma = state[0]
    return np.array(signal_probs), np.array(garbage_probs), p_sigma


def _get_target_phases(depth, target_name):
    """Compute QSP phases for a target polynomial of given degree.

    Returns (phases, target_label, target_func).
    Falls back to analytically motivated phases when pyqsp is unavailable.
    """
    deg = depth
    if deg % 2 != 0:
        deg += 1  # Ensure even degree for clean parity

    if not _HAS_QSP:
        # Analytically motivated fallback: alternating ±π/4 phases implement
        # a Chebyshev-like polynomial that shows clear phase-steering effects.
        # The key pedagogical point is that ANY non-trivial phases create a
        # different polynomial from the no-phase case.
        n_phases = deg + 1
        phases = np.zeros(n_phases)
        phases[0] = np.pi / 4
        phases[-1] = -np.pi / 4
        for j in range(1, n_phases - 1):
            phases[j] = (-1) ** j * np.pi / (4 + j)
        return phases, f"{target_name} (demo phases — install pyqsp for exact)", None

    try:
        if target_name == "x^k (matrix power)":
            k = max(2, deg // 2)
            target_f = lambda x: np.clip(x ** k, -1 + 1e-5, 1 - 1e-5)
            label = f"$x^{{{k}}}$ (matrix power)"
        elif target_name == "sign(x) (step function)":
            target_f = lambda x: np.clip(
                np.tanh(8 * x), -1 + 1e-5, 1 - 1e-5
            )
            label = r"$\mathrm{sign}(x)$ (smooth step)"
        else:  # exp(-t|x|-1) (diffusion)
            k = max(2, deg // 2)
            target_f = lambda x: np.exp(k * (np.abs(x) - 1))
            label = f"$e^{{{k}(|x|-1)}}$ (diffusion)"

        coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-4, npts=800)
        phases = Angles_Fixed(coef, tolerance=1e-4)
        return phases, label, target_f
    except Exception:
        n_phases = deg + 1
        phases = np.zeros(n_phases)
        phases[0] = np.pi / 4
        phases[-1] = -np.pi / 4
        for j in range(1, n_phases - 1):
            phases[j] = (-1) ** j * np.pi / (4 + j)
        return phases, f"{target_name} (demo phases)", None


def _draw_qsp_comparison(depth, frame, sigma, phases):
    """Draw the true 2×2 QSP comparison: naive (no phases) vs interleaved."""
    sig_inter, garb_inter, p_sigma = _simulate_qsp(sigma, phases, use_phases=True)
    sig_naive, garb_naive, p_naive = _simulate_qsp(sigma, phases, use_phases=False)

    k = min(frame + 1, len(phases))

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.35, wspace=0.30)

    # ── Top panels: state trajectory on unit circle ──
    for col_idx, (label, sig_p, garb_p, color, p_val) in enumerate([
        ("Naive: all $\\phi_j = 0$", sig_naive, garb_naive, "#c62828", p_naive),
        ("Interleaved: true QSP phases", sig_inter, garb_inter, "#2e7d32", p_sigma),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        # Draw unit circle background
        theta = np.linspace(0, np.pi / 2, 100)
        ax.plot(np.sin(theta), np.cos(theta), '--', color='#ccc', lw=1.5)
        ax.fill_between(np.sin(theta), np.cos(theta), alpha=0.03, color='gray')

        # Plot trajectory up to current frame
        s_vals = np.sqrt(np.clip(sig_p[:k + 1], 0, 1))
        g_vals = np.sqrt(np.clip(garb_p[:k + 1], 0, 1))

        ax.plot(g_vals, s_vals, '-', color=color, lw=2.5, alpha=0.8)

        # Step markers with color gradient
        for i in range(k + 1):
            frac = i / max(k, 1)
            marker_color = plt.cm.YlOrRd(0.2 + 0.7 * frac) if col_idx == 0 else plt.cm.YlGn(0.2 + 0.7 * frac)
            ax.plot(g_vals[i], s_vals[i], 'o', color=marker_color, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
            ax.annotate(str(i), (g_vals[i], s_vals[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.7)

        # Start and end markers
        ax.plot(g_vals[0], s_vals[0], 's', color='blue', markersize=10,
                markeredgecolor='black', zorder=6, label="Start")
        ax.plot(g_vals[k], s_vals[k], 'X', color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1.2, zorder=6, label="Current")

        # Labels
        ax.set_xlabel(r"$|\langle\perp|\psi\rangle|$ (garbage)", fontsize=11, fontweight='bold')
        ax.set_ylabel(r"$|\langle 0|\psi\rangle|$ (signal)", fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold', color=color, pad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.2)

        # Stats box
        final_sig = sig_p[k]
        final_garb = garb_p[k]
        ax.text(0.98, 0.98,
                f"$|\\langle 0|\\psi\\rangle|^2$ = {final_sig:.4f}\n"
                f"$|\\langle\\perp|\\psi\\rangle|^2$ = {final_garb:.4f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.85, edgecolor='#bbb'))

    # ── Bottom panel: signal & garbage evolution over steps ──
    ax_bottom = fig.add_subplot(gs[1, :])
    steps = np.arange(k + 1)

    ax_bottom.plot(steps, sig_naive[:k + 1], 's--', color='#c62828', lw=1.8,
                   markersize=5, label='Signal (naive)', alpha=0.8)
    ax_bottom.plot(steps, garb_naive[:k + 1], 'o--', color='#ff9800', lw=1.5,
                   markersize=4, label='Garbage (naive)', alpha=0.7)
    ax_bottom.plot(steps, sig_inter[:k + 1], 's-', color='#2e7d32', lw=2.2,
                   markersize=5, label='Signal (interleaved)', alpha=0.9)
    ax_bottom.plot(steps, garb_inter[:k + 1], 'o-', color='#81c784', lw=1.5,
                   markersize=4, label='Garbage (interleaved)', alpha=0.7)
    ax_bottom.axhline(y=1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax_bottom.set_xlabel("QSP step", fontsize=11, fontweight='bold')
    ax_bottom.set_ylabel("Probability", fontsize=11, fontweight='bold')
    ax_bottom.set_title("Signal retention vs garbage accumulation", fontsize=11, fontweight='bold')
    ax_bottom.legend(fontsize=8, ncol=4, loc='upper center',
                     bbox_to_anchor=(0.5, -0.15))
    ax_bottom.set_xlim(-0.3, max(k, 1) + 0.3)
    ax_bottom.set_ylim(-0.05, 1.1)
    ax_bottom.grid(True, alpha=0.2)

    # Final P(σ) annotation
    p_val_display = abs(p_sigma) ** 2
    fig.suptitle(
        f"True QSP on 2×2 subspace | σ = {sigma:.2f} | "
        f"Final $|P(\\sigma)|^2$ = {p_val_display:.4f} (interleaved), "
        f"{abs(p_naive)**2:.4f} (naive)",
        fontsize=12, fontweight='bold', y=0.99
    )
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.12, top=0.92)
    return fig


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
(SVD factorises any matrix $A = U\Sigma V^\dagger$ where $\Sigma$ is diagonal with entries
$\sigma_i \ge 0$.  Think of it as: rotate input basis ($V^\dagger$), scale each component
($\Sigma$), rotate output basis ($U$).  QSVT transforms each $\sigma_i \to P(\sigma_i)$
independently.)

In each two-dimensional signal/garbage invariant subspace (single-ancilla picture),
$U_A$ mixes ancilla signal $|0\rangle$ and garbage $|\perp\rangle$ by an amount that
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

where $P$ is a **degree-$d$ polynomial** that we control. In the ideal synthesis
model, garbage paths destructively interfere; in finite-precision implementations,
small residual contamination can remain.

> **Precision note:** The QSVT circuit implements the polynomial $P$ to machine precision,
> but $P$ is itself a Chebyshev *approximation* of the target function $f$ to within $\varepsilon$.
""")

    st.caption(
        "Mental model: QSVT is phase-engineered interference on ancilla-conditioned paths. "
        "The polynomial is the resulting transfer function on singular values."
    )

    st.markdown("---")
    st.markdown("### Animated Intuition: Post-Selection & Phase Steering")

    tab_exact, tab_schematic = st.tabs(["Exact QSP Model (2×2)", "Schematic Analogy (3D)"])

    # ── Tab 1: True 2×2 QSP model ──────────────────────────────────────
    with tab_exact:
        st.caption(
            "This uses the **real QSP math**: for a single singular value σ, "
            "the block encoding W(σ) and phase gates D(φ) act on a 2D subspace "
            "{|0⟩, |⊥⟩}. Every number shown is an exact matrix product — no fudge factors."
        )

        ec1, ec2, ec3 = st.columns([1, 1, 1])
        with ec1:
            qsp_depth = st.slider("Degree (# of QSP steps)", 2, 20, 8, 2, key="s04_qsp_depth")
        with ec2:
            qsp_sigma = st.slider("Singular value σ", 0.10, 0.99, 0.75, 0.01, key="s04_qsp_sigma")
        with ec3:
            target_name = st.selectbox(
                "Target polynomial",
                ["x^k (matrix power)", "sign(x) (step function)", "exp(-t|x|-1) (diffusion)"],
                key="s04_qsp_target"
            )

        # Compute or cache phases
        cache_key = f"s04_qsp_phases_{qsp_depth}_{target_name}"
        if cache_key not in st.session_state:
            with st.spinner("Computing QSP phases..."):
                phases, label, _ = _get_target_phases(qsp_depth, target_name)
                st.session_state[cache_key] = (phases, label)
        phases, label = st.session_state[cache_key]

        if not _HAS_QSP:
            st.warning("pyqsp not available — using heuristic phase schedule. "
                       "Install pyqsp for exact QSP angles.")

        st.markdown(f"**Target:** {label} &nbsp;|&nbsp; **Phases:** {len(phases)} angles")

        ef_col, ep_col = st.columns([2, 1])
        with ef_col:
            qsp_frame = st.slider("Current step", 0, len(phases) - 1,
                                  len(phases) - 1, 1, key="s04_qsp_frame")
        with ep_col:
            qsp_play = st.button("Play ▶", key="s04_qsp_play", use_container_width=True)

        qsp_ph = st.empty()
        if qsp_play:
            for fr in range(len(phases)):
                fig_qsp = _draw_qsp_comparison(qsp_depth, fr, qsp_sigma, phases)
                qsp_ph.pyplot(fig_qsp, use_container_width=True)
                plt.close(fig_qsp)
                time.sleep(0.15)
        else:
            fig_qsp = _draw_qsp_comparison(qsp_depth, qsp_frame, qsp_sigma, phases)
            qsp_ph.pyplot(fig_qsp, use_container_width=True)
            plt.close(fig_qsp)

        st.info(
            "**What you see:** Each point is an exact 2×2 matrix product. "
            "The signal axis shows |⟨0|ψ⟩|, the garbage axis shows |⟨⊥|ψ⟩|. "
            "Without phases (naive), the state wanders into the garbage direction. "
            "With the correct QSP phases, destructive interference keeps the state "
            "near the signal axis, and the final |⟨0|ψ⟩|² = |P(σ)|²."
        )

        with st.expander("The 2×2 math behind this plot"):
            st.markdown(r"""
For a single singular value $\sigma$, the block encoding acts on the 2D subspace $\{|0\rangle, |\perp\rangle\}$:

$$W(\sigma) = \begin{pmatrix} \sigma & \sqrt{1-\sigma^2} \\ -\sqrt{1-\sigma^2} & \sigma \end{pmatrix}, \qquad D(\phi) = \begin{pmatrix} e^{i\phi} & 0 \\ 0 & e^{-i\phi} \end{pmatrix}$$

The QSP sequence alternates phase gates and block encodings:

$$U_{\mathrm{QSP}} = D(\phi_0)\, W\, D(\phi_1)\, W^\dagger\, D(\phi_2)\, W \cdots$$

Starting from $|\psi_0\rangle = |0\rangle = (1, 0)^T$, the final state is:

$$|\psi_d\rangle = U_{\mathrm{QSP}} |0\rangle = \begin{pmatrix} P(\sigma) \\ Q(\sigma)\sqrt{1-\sigma^2} \end{pmatrix}$$

where $|P(\sigma)|^2 + |Q(\sigma)|^2(1-\sigma^2) = 1$ (unitarity).
The top entry $P(\sigma)$ is the desired polynomial — this is QSVT.
""")

    # ── Tab 2: Schematic 3D analogy ─────────────────────────────────────
    with tab_schematic:
        st.caption(
            "**Qualitative analogy** (toy model for visual intuition): "
            "(x, y) is the relevant signal plane, z is out-of-plane garbage. "
            "Each $U_A$ rotates by $\\theta$ in-plane and leaks by $\\alpha$ out-of-plane; "
            "post-selection projects back to z = 0. "
            "For the exact math, see the **Exact QSP Model** tab."
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
        "**The QSVT insight:** Every query rotates the useful state in-plane but also leaks amplitude out-of-plane. "
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

    st.markdown("---")

    # ── Grand unification table ──
    st.markdown("### Algorithms Unified by QSVT")

    st.markdown(r"""
| Algorithm | Polynomial $P(\sigma)$ | Degree | Application |
|-----------|----------------------|--------|-------------|
| **Hamiltonian simulation** | $e^{-it\sigma}$ | $O(t + \log(1/\epsilon))$ | Time evolution |
| **Matrix inversion (HHL)** | $1/\sigma$ | $O(\kappa_{\mathrm{cond}}/\epsilon)$ | Linear systems |
| **Amplitude amplification** | Chebyshev of $\sigma$ | $O(1/\sqrt{p})$ | Search |
| **Phase estimation** | Step function | $O(1/\epsilon)$ | Eigenvalues |
| **Quantum walks** | $\sigma \mapsto e^{i\arccos\sigma}$ | $O(1)$ | Graph problems |
""")

    st.caption(
        r"Here $p$ is initial success probability, $\kappa_{\mathrm{cond}} = \alpha/\sigma_{\min}$ is the condition number "
        r"($\sigma_{\min}$ = smallest singular value), and $\epsilon$ is approximation error."
    )

    key_concept(
        "QSVT transforms each singular value $\\sigma_i \\mapsto P(\\sigma_i)$ by interleaving "
        "a block encoding with phase rotations. The phases create <b>destructive interference</b> "
        "that eliminates garbage cross-terms — giving an exact polynomial transformation in "
        "$O(d)$ queries. Different polynomials $P$ yield different quantum algorithms."
    )

    reference_list(["Martyn2021", "Gilyen2019", "Low2017"])
