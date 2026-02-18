"""Slide 7b: Live Demo — 2D Advection-Diffusion PDE."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from slides.components import slide_header

TITLE = "Demo: 2D PDE Solver"


def render():
    slide_header("Live Demo — 2D Advection-Diffusion",
                 "5-point stencil Laplacian + QFT advection on a periodic 2-D grid")

    from simulation import exact_solution_fourier_2d, get_classical_matrix_2d
    from quantum import QSVT_circuit_2d, Advection_Gate_2d
    from solvers import cvx_poly_coef, Angles_Fixed

    # ── Controls ──────────────────────────────────────────────────────────
    st.markdown("### Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        nx = st.slider("Grid width (nx qubits)", 2, 6, 3, 1, key="d2d_nx")
        ny = st.slider("Grid height (ny qubits)", 2, 6, 3, 1, key="d2d_ny")
        n_total = nx + ny
        st.caption(f"{n_total} data qubits, {2**nx} x {2**ny} grid")
    with c2:
        nu = st.slider("Viscosity", 0.005, 0.05, 0.02, 0.001, key="d2d_nu")
        cx = st.slider("X-velocity", -1.0, 1.0, 0.3, 0.05, key="d2d_cx")
        cy = st.slider("Y-velocity", -1.0, 1.0, 0.3, 0.05, key="d2d_cy")
    with c3:
        t_max = st.slider("Max steps", 10, 60, 30, 10, key="d2d_tmax")
        n_snap = st.slider("Snapshots", 3, 8, 4, 1, key="d2d_snap")
    with c4:
        ic_type = st.selectbox("Initial condition", [
            "Gaussian Peak", "Double Gaussian", "Sine Pattern",
        ], key="d2d_ic")
        shots = st.number_input("Shots", 10000, 200000, 100000, 10000, key="d2d_shots")

    if n_total > 12:
        st.error(f"Total qubits ({n_total}) exceeds safe limit. Reduce grid size.")
        return

    Nx, Ny = 2 ** nx, 2 ** ny
    time_steps = [int(i * t_max / (n_snap - 1)) for i in range(n_snap)]

    # ── Initial condition ─────────────────────────────────────────────────
    x_arr = np.linspace(0, 1, Nx, endpoint=False)
    y_arr = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")

    if ic_type == "Gaussian Peak":
        u0_2d = np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    elif ic_type == "Double Gaussian":
        u0_2d = (np.exp(-8 * ((X - 0.3) ** 2 + (Y - 0.3) ** 2))
                 + np.exp(-8 * ((X - 0.7) ** 2 + (Y - 0.7) ** 2)))
    else:
        u0_2d = np.abs(np.sin(2 * np.pi * 2 * X) * np.sin(2 * np.pi * 2 * Y))

    norm_val = np.linalg.norm(u0_2d.flatten())
    if norm_val > 0:
        u0_2d_n = u0_2d / norm_val
    else:
        u0_2d_n = u0_2d

    # ── Preview ───────────────────────────────────────────────────────────
    fig_p, ax_p = plt.subplots(figsize=(4, 3.5))
    from scipy.ndimage import zoom
    u0_fine = zoom(u0_2d_n, (max(1, 128 // Nx), max(1, 128 // Ny)), order=1)
    ax_p.imshow(u0_fine, cmap="viridis", origin="lower",
                extent=[0, 1, 0, 1], interpolation="bilinear")
    ax_p.set_title("Initial condition", fontsize=11, fontweight="bold")
    ax_p.set_xlabel("X"); ax_p.set_ylabel("Y")
    fig_p.tight_layout()
    st.pyplot(fig_p, use_container_width=True)
    plt.close(fig_p)

    st.markdown("---")

    # ── Run simulation ────────────────────────────────────────────────────
    run_btn = st.button("Run 2-D Quantum Simulation", type="primary", key="d2d_run")

    if run_btn:
        A_matrix, dt = get_classical_matrix_2d(nx, ny, nu, cx, cy)

        full_dim = 2 ** n_total
        u0_flat = u0_2d.flatten()
        u0_full = np.zeros(full_dim)
        u0_full[:len(u0_flat)] = u0_flat
        u0_full /= np.linalg.norm(u0_full)

        n_cols = min(3, len(time_steps))
        n_rows = (len(time_steps) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        status = st.empty()
        pbar = st.progress(0)
        plot_ph = st.empty()

        v_cl = u0_flat.copy()
        cur = 0

        for si, t in enumerate(sorted(time_steps)):
            ri, ci_ax = si // n_cols, si % n_cols
            ax = axes[ri, ci_ax]
            status.markdown(f"**Computing step {t} …**")

            if t == 0:
                u_fine = zoom(u0_2d_n, (max(1, 128 // Nx), max(1, 128 // Ny)), order=1)
                ax.imshow(u_fine, cmap="viridis", origin="lower",
                          extent=[0, 1, 0, 1], interpolation="bilinear")
                ax.set_title("t = 0 (Initial)", fontsize=11, fontweight="bold")
                ax.set_xlabel("X"); ax.set_ylabel("Y")
                pbar.progress((si + 1) / len(time_steps))
                continue

            deg = int(t + 8)
            if deg % 2 != 0:
                deg += 1

            try:
                target_f = lambda x, _t=t: np.exp(_t * (np.abs(x) - 1))
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)

                qc = QSVT_circuit_2d(phi_seq, nx, ny, nu,
                                     init_state=u0_full, measurement=True)

                phys_t = t * dt
                adv_gate = Advection_Gate_2d(nx, ny, cx, cy, phys_t)
                x_reg = y_reg = None
                for qreg in qc.qregs:
                    if qreg.name == "x":
                        x_reg = qreg
                    elif qreg.name == "y":
                        y_reg = qreg
                if x_reg and y_reg:
                    qc.append(adv_gate, list(x_reg) + list(y_reg))

                from qiskit import transpile
                from qiskit_aer import AerSimulator

                backend = AerSimulator()
                tqc = transpile(qc, backend, optimization_level=0)
                counts = backend.run(tqc, shots=shots).result().get_counts()

                prob_dist = np.zeros(Nx * Ny)
                total_valid = 0
                for bs, cnt in counts.items():
                    parts = bs.split()
                    if len(parts) != 3:
                        continue
                    m_dat, m_anc, m_sig = parts
                    if (len(m_dat) == n_total and len(m_anc) == 2
                            and len(m_sig) == 1
                            and m_sig == "0" and m_anc == "00"):
                        x_idx = int(m_dat[:nx], 2)
                        y_idx = int(m_dat[nx:], 2)
                        flat_idx = x_idx * Ny + y_idx
                        if 0 <= flat_idx < Nx * Ny:
                            prob_dist[flat_idx] += cnt
                            total_valid += cnt

                if total_valid > 0:
                    prob_dist /= total_valid
                    yq_2d = np.sqrt(prob_dist).reshape((Nx, Ny))
                    yq_fine = zoom(yq_2d,
                                   (max(1, 128 // Nx), max(1, 128 // Ny)), order=1)
                    ax.imshow(yq_fine, cmap="viridis", origin="lower",
                              extent=[0, 1, 0, 1], interpolation="bilinear")
                    ax.set_title(
                        f"t={t}  ({100*total_valid/shots:.0f}% valid)",
                        fontsize=11, fontweight="bold")
                else:
                    ax.text(0.5, 0.5, "No valid counts", transform=ax.transAxes,
                            ha="center", va="center", color="red")
                ax.set_xlabel("X"); ax.set_ylabel("Y")
            except Exception as e:
                ax.text(0.5, 0.5, str(e)[:40], transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="red")

            pbar.progress((si + 1) / len(time_steps))

        # hide leftover axes
        for extra in range(si + 1, n_rows * n_cols):
            axes[extra // n_cols, extra % n_cols].axis("off")

        fig.tight_layout()
        plot_ph.pyplot(fig)
        plt.close(fig)
        status.markdown("**2-D simulation complete.**")
        pbar.progress(1.0)
