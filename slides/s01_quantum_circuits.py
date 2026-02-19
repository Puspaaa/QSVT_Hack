"""Slide 1: Quantum Circuits Refresher — smooth animation + measurement collapse."""

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from slides.components import slide_header, reference, key_concept

TITLE = "Quantum Circuits Refresher"

# ── Gate definitions ──────────────────────────────────────────────────────
_ry_ang = np.pi / 3  # 60-degree Y-rotation

GATES = {
    "H":  np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "T":  np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    "Ry": np.array([
        [ np.cos(_ry_ang / 2), -np.sin(_ry_ang / 2)],
        [ np.sin(_ry_ang / 2),  np.cos(_ry_ang / 2)],
    ]),
}

STEPS = [
    ("|0>", None, "Initial state — north pole"),
    ("H",   "H",  "Hadamard — equal superposition"),
    ("T",   "T",  "T gate — phase rotation (pi/4)"),
    ("Ry",  "Ry", "Ry(60) — tilts state off equator"),
]

CIRCUIT_LABELS = ["|0>", "H", "T", "Ry", "M"]
CIRCUIT_XS     = [0.5, 2.0, 3.5, 5.0, 6.8]

INTERP_FRAMES = 8
FRAME_DELAY   = 0.12
HOLD_DELAY    = 0.5
MEAS_FRAMES   = 6
MEAS_DELAY    = 0.18


# ── Math helpers ──────────────────────────────────────────────────────────

def _bloch_angles(psi):
    theta = 2 * np.arccos(np.clip(np.abs(psi[0]), 0.0, 1.0))
    phi = (np.angle(psi[1]) - np.angle(psi[0])) if np.abs(psi[0]) > 1e-10 else np.angle(psi[1])
    return theta, phi

def _bloch_xyz(psi):
    th, ph = _bloch_angles(psi)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])

def _xyz_to_angles(v):
    r = np.linalg.norm(v)
    if r < 1e-12: return 0.0, 0.0
    v = v / r
    return np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0])

def _slerp(v0, v1, t):
    v0n = v0 / (np.linalg.norm(v0) + 1e-15)
    v1n = v1 / (np.linalg.norm(v1) + 1e-15)
    dot = np.clip(np.dot(v0n, v1n), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < 1e-8:
        return v0n + t * (v1n - v0n)
    return (np.sin((1-t)*omega)*v0n + np.sin(t*omega)*v1n) / np.sin(omega)

def _fmt(z, tol=1e-10):
    r, i = np.real(z), np.imag(z)
    if abs(i) < tol:
        return str(int(round(r))) if abs(r - round(r)) < tol else f"{r:.2f}"
    if abs(r) < tol:
        if abs(i-1) < tol: return "i"
        if abs(i+1) < tol: return "-i"
        return f"{i:.2f}i"
    sign = "+" if i > 0 else "-"
    return f"{r:.2f}{sign}{abs(i):.2f}i"


# ── Bloch sphere with multiple arrows ────────────────────────────────────

def _draw_bloch(arrows, figsize=(3.5, 3.5)):
    """arrows: list of (theta, phi, color, alpha, linewidth)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ax.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color='lightblue', alpha=0.15, linewidth=0.4)
    ax.plot([-1.3,1.3],[0,0],[0,0],'k-',alpha=0.2,lw=0.5)
    ax.plot([0,0],[-1.3,1.3],[0,0],'k-',alpha=0.2,lw=0.5)
    ax.plot([0,0],[0,0],[-1.3,1.3],'k-',alpha=0.2,lw=0.5)
    ax.text(0,0, 1.45,r'$|0\rangle$',fontsize=12,ha='center')
    ax.text(0,0,-1.45,r'$|1\rangle$',fontsize=12,ha='center')
    for th, ph, c, a, lw in arrows:
        if a < 0.02:
            continue
        sx = np.sin(th)*np.cos(ph)
        sy = np.sin(th)*np.sin(ph)
        sz = np.cos(th)
        ax.quiver(0,0,0, sx,sy,sz, color=c, alpha=float(np.clip(a,0,1)),
                  arrow_length_ratio=0.12, linewidth=lw)
    ax.set_xlim([-1.5,1.5]); ax.set_ylim([-1.5,1.5]); ax.set_zlim([-1.5,1.5])
    ax.set_axis_off(); fig.tight_layout()
    return fig


# ── Circuit diagram with loading-bar fill ────────────────────────────────

def _draw_circuit(active_idx, fill_frac):
    """active_idx: 0=|0>, 1=H, 2=T, 3=Ry, 4=M.  fill_frac: 0..1."""
    box_w, box_h = 0.8, 0.6
    fig, ax = plt.subplots(figsize=(8, 0.85))
    ax.set_xlim(-0.2, 7.8); ax.set_ylim(-0.55, 0.55); ax.axis("off")
    ax.plot([0.0, 7.5], [0, 0], color="#888", lw=2, zorder=0)

    for i, (lbl, x) in enumerate(zip(CIRCUIT_LABELS, CIRCUIT_XS)):
        if i == 0:
            done = active_idx >= 1
            ax.text(x,0,lbl,ha="center",va="center",fontsize=14,
                    fontweight="bold" if done else "normal",
                    color="#1a73e8" if done else "#666",zorder=2)
            continue

        left, bot = x - box_w/2, -box_h/2

        if i < active_idx:
            r = mpatches.FancyBboxPatch((left,bot),box_w,box_h,
                    boxstyle="round,pad=0.08",facecolor="#1a73e8",
                    edgecolor="#0d47a1",lw=2,zorder=1)
            ax.add_patch(r)
            ax.text(x,0,lbl,ha="center",va="center",fontsize=14,
                    fontweight="bold",color="white",zorder=3)

        elif i == active_idx:
            bg = mpatches.FancyBboxPatch((left,bot),box_w,box_h,
                    boxstyle="round,pad=0.08",facecolor="#e0e0e0",
                    edgecolor="#0d47a1",lw=2.5,zorder=1)
            ax.add_patch(bg)
            bar = mpatches.Rectangle((left,bot),box_w*fill_frac,box_h,
                    facecolor="#1a73e8",alpha=0.85,zorder=1.5)
            bar.set_clip_path(bg)
            ax.add_patch(bar)
            ax.text(x,0,lbl,ha="center",va="center",fontsize=14,
                    fontweight="bold",color="white",zorder=3)

        else:
            r = mpatches.FancyBboxPatch((left,bot),box_w,box_h,
                    boxstyle="round,pad=0.08",facecolor="#e8eaf6",
                    edgecolor="#90a4ae",lw=1.5,zorder=1)
            ax.add_patch(r)
            ax.text(x,0,lbl,ha="center",va="center",fontsize=14,
                    fontweight="bold",color="#333",zorder=3)

    fig.tight_layout(pad=0.1)
    return fig


# ── Frame renderers ──────────────────────────────────────────────────────

def _frame(container, active_idx, fill, desc, arrows, psi_d):
    with container.container():
        fig_c = _draw_circuit(active_idx, fill)
        st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)
        c1, c2 = st.columns([1, 1])
        p = np.abs(psi_d)**2
        with c1:
            st.markdown(f"#### {desc}")
            st.latex(
                rf"|\psi\rangle = {_fmt(psi_d[0])}\,|0\rangle"
                rf" + {_fmt(psi_d[1])}\,|1\rangle")
            st.markdown(f"**P(0) = {p[0]:.0%}** &nbsp;&nbsp; **P(1) = {p[1]:.0%}**")
        with c2:
            fig_b = _draw_bloch(arrows)
            st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)


# ──────────────────────────────────────────────────────────────────────────

def render():
    slide_header("Quantum Circuits: A Quick Refresher",
                 "Framing quantum computation for physicists")

    st.markdown(
        r"A quantum circuit applies **unitary gates** to qubits. "
        r"Each gate is a **rotation** of the Bloch vector. "
        r"Measurement collapses the state with probability "
        r"$p_j = |\langle j|\psi\rangle|^2$."
    )

    # ── Precompute states ────────────────────────────────────────────────
    psi = np.array([1, 0], dtype=complex)
    ep = [psi.copy()]
    for _, gk, _ in STEPS[1:]:
        psi = GATES[gk] @ psi
        ep.append(psi.copy())
    ep_v = [_bloch_xyz(s) for s in ep]

    frame = st.empty()

    # Step 0 — hold |0⟩
    th0, ph0 = _bloch_angles(ep[0])
    _frame(frame, 0, 0, STEPS[0][2],
           [(th0, ph0, '#7b61ff', 1.0, 2.5)], ep[0])
    time.sleep(HOLD_DELAY)

    # Steps 1-3 — smooth gate rotations
    for si in range(1, len(STEPS)):
        _, _, desc = STEPS[si]
        v0, v1 = ep_v[si-1], ep_v[si]
        for f in range(INTERP_FRAMES + 1):
            t = f / INTERP_FRAMES
            vi = _slerp(v0, v1, t)
            thi, phi = _xyz_to_angles(vi)
            psi_d = (1-t)*ep[si-1] + t*ep[si]
            n = np.linalg.norm(psi_d)
            if n > 1e-12: psi_d /= n
            _frame(frame, si, t, desc,
                   [(thi, phi, '#7b61ff', 1.0, 2.5)], psi_d)
            if f < INTERP_FRAMES:
                time.sleep(FRAME_DELAY)
        time.sleep(HOLD_DELAY)

    # ── Measurement animation ────────────────────────────────────────────
    final = ep[-1]
    fth, fph = _bloch_angles(final)
    pr = np.abs(final)**2

    # Collapse to the more-probable outcome
    col_to = int(np.argmax(pr))
    col_th = np.pi * col_to     # 0 → north, 1 → south
    lose_th = np.pi * (1 - col_to)
    win_c  = '#e74c3c' if col_to == 1 else '#2ecc71'
    lose_c = '#2ecc71' if col_to == 1 else '#e74c3c'
    col_psi = np.array([1, 0], dtype=complex) if col_to == 0 else np.array([0, 1], dtype=complex)

    # Phase A — split: main vector fades, two ghost arrows grow to poles
    for f in range(MEAS_FRAMES + 1):
        t = f / MEAS_FRAMES
        arrows = [
            (fth,  fph, '#7b61ff', max(1.0 - t, 0.1), 2.5),
            (0.0,  0.0, '#2ecc71', t * pr[0] * 1.4, 2.0),
            (np.pi,0.0, '#e74c3c', t * pr[1] * 1.4, 2.0),
        ]
        _frame(frame, 4, t * 0.5,
               "Measurement — superposition splitting...", arrows, final)
        if f < MEAS_FRAMES:
            time.sleep(MEAS_DELAY)
    time.sleep(0.3)

    # Phase B — collapse: loser fades, winner solidifies
    for f in range(MEAS_FRAMES + 1):
        t = f / MEAS_FRAMES
        arrows = [
            (col_th,  0.0, win_c,  0.5 + 0.5*t, 2.0 + 1.5*t),
            (lose_th, 0.0, lose_c, max(0.9*(1-t), 0.0), max(2*(1-t), 0.1)),
        ]
        psi_d = (1-t)*final + t*col_psi
        n = np.linalg.norm(psi_d)
        if n > 1e-12: psi_d /= n
        lbl = f"Collapsed to |{col_to}>" if t >= 0.5 else "Collapsing..."
        _frame(frame, 4, 0.5 + t*0.5, lbl, arrows, psi_d)
        if f < MEAS_FRAMES:
            time.sleep(MEAS_DELAY)
    time.sleep(HOLD_DELAY)

    # ── Key concept ──────────────────────────────────────────────────────
    key_concept(
        "A quantum circuit = a <b>product of unitaries</b>. "
        "Each gate rotates the Bloch vector; measurement collapses it "
        "to $|0\\rangle$ or $|1\\rangle$ with probability "
        "$|\\langle j|\\psi\\rangle|^2$."
    )

    reference("NC2000")
