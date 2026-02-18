"""Slide 6b: Live Demo — 1D Advection-Diffusion PDE."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from slides.components import slide_header

TITLE = "Demo: 1D PDE Solver"


def render():
    slide_header("Live Demo — 1D Advection-Diffusion",
                 "QSVT diffusion + QFT advection on a periodic domain")

    from simulation import run_split_step_sim, exact_solution_fourier, get_classical_matrix
    from quantum import QSVT_circuit_universal, Block_encoding_diffusion, Advection_Gate
    from solvers import cvx_poly_coef, Angles_Fixed

    # ── Controls ──────────────────────────────────────────────────────────
    st.markdown("### Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_qubits = st.slider("Qubits", 3, 8, 5, 1, key="d1d_n")
        st.caption(f"Grid N = {2**n_qubits}")
    with c2:
        nu = st.slider("Viscosity", 0.005, 0.05, 0.02, 0.001, key="d1d_nu")
        c_vel = st.slider("Advection vel.", -1.0, 1.0, 0.5, 0.05, key="d1d_c")
    with c3:
        t_max = st.slider("Max steps", 10, 100, 30, 10, key="d1d_tmax")
        n_snap = st.slider("Snapshots", 3, 12, 5, 1, key="d1d_snap")
    with c4:
        ic_type = st.selectbox("Initial condition", [
            "Gaussian Peak", "Double Gaussian", "Sine Wave",
        ], key="d1d_ic")
        shots = st.number_input("Shots", 10000, 500000, 100000, 50000, key="d1d_shots")

    time_steps = [int(i * t_max / (n_snap - 1)) for i in range(n_snap)]

    # ── Initial condition ─────────────────────────────────────────────────
    if ic_type == "Gaussian Peak":
        u0_func = lambda x: np.exp(-100 * (x - 0.3) ** 2)
    elif ic_type == "Double Gaussian":
        u0_func = lambda x: (np.exp(-80 * (x - 0.25) ** 2)
                              + np.exp(-80 * (x - 0.75) ** 2))
    else:
        u0_func = lambda x: np.abs(np.sin(2 * np.pi * 2 * x))

    N = 2 ** n_qubits
    x_grid = np.linspace(0, 1, N, endpoint=False)
    u0_vals = u0_func(x_grid)

    # ── Preview ───────────────────────────────────────────────────────────
    col_prev, col_circ = st.columns([1, 1])
    with col_prev:
        fig_p, ax_p = plt.subplots(figsize=(5, 2.5))
        y_init = u0_vals / np.linalg.norm(u0_vals)
        ax_p.plot(x_grid, y_init, "b-", lw=2)
        ax_p.fill_between(x_grid, y_init, alpha=0.2)
        ax_p.set_xlabel("x"); ax_p.set_ylabel("u(x,0)")
        ax_p.set_title("Initial condition (normalized)", fontsize=11, fontweight="bold")
        ax_p.grid(True, alpha=0.3)
        fig_p.tight_layout()
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

    with col_circ:
        st.markdown("**Block-encoding circuit (LCU)**")
        try:
            qc_block = Block_encoding_diffusion(n_qubits, nu)
            w = max(12, n_qubits * 2)
            h = max(3, n_qubits * 0.5 + 1.5)
            fig_c, ax_c = plt.subplots(figsize=(w, h), dpi=120)
            qc_block.draw("mpl", fold=-1, ax=ax_c, style={
                "backgroundcolor": "#FFFFFF", "gatefacecolor": "#BB8FCE"})
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
        except Exception as exc:
            st.info(f"Circuit drawing skipped: {exc}")

    st.markdown("---")

    # ── Run simulation ────────────────────────────────────────────────────
    run_btn = st.button("Run 1-D Quantum Simulation", type="primary", key="d1d_run")

    if run_btn:
        A_matrix, dt = get_classical_matrix(N, nu, c_vel)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps)))
        y0 = u0_vals / np.linalg.norm(u0_vals)
        ax.plot(x_grid, y0, color=colors[0], lw=2.5, label="t = 0 (initial)")
        ax.fill_between(x_grid, y0, alpha=0.15, color=colors[0])
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_title(
            f"1-D Advection-Diffusion  (n={n_qubits}, v={nu}, c={c_vel})",
            fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plot_ph = st.empty()
        plot_ph.pyplot(fig)
        status = st.empty()
        pbar = st.progress(0)

        v_cl = u0_vals.copy()
        cur = 0
        steps = sorted(time_steps)

        for idx, t in enumerate(steps[1:], 1):
            status.markdown(f"**Computing time-step {t} …**")
            deg = int(t + 8)
            if deg % 2 != 0:
                deg += 1
            try:
                target_f = lambda x, _t=t: np.exp(_t * (np.abs(x) - 1))
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)

                from qiskit import transpile
                from qiskit_aer import AerSimulator

                state_vec = u0_vals / np.linalg.norm(u0_vals)
                qc = QSVT_circuit_universal(phi_seq, n_qubits, nu,
                                            init_state=state_vec, measurement=False)
                phys_t = t * dt
                qc.append(Advection_Gate(n_qubits, c_vel, phys_t), qc.qregs[2])
                qc.measure(qc.qregs[0], qc.cregs[0])
                qc.measure(qc.qregs[1], qc.cregs[1])
                qc.measure(qc.qregs[2], qc.cregs[2])

                backend = AerSimulator()
                tqc = transpile(qc, backend, optimization_level=0)
                counts = backend.run(tqc, shots=shots).result().get_counts()

                prob_dist = np.zeros(N)
                total_valid = 0
                for key, cnt in counts.items():
                    parts = key.split()
                    s_bit, a_bits, d_bits = "", "", ""
                    for p in parts:
                        if len(p) == 1:
                            s_bit = p
                        elif len(p) == 2:
                            a_bits = p
                        elif len(p) == n_qubits:
                            d_bits = p
                    if s_bit == "0" and a_bits == "00":
                        prob_dist[int(d_bits, 2)] += cnt
                        total_valid += cnt

                if total_valid > 0:
                    prob_dist /= total_valid
                    yq = np.sqrt(prob_dist)

                    gap = t - cur
                    if gap > 0:
                        v_cl = np.linalg.matrix_power(A_matrix, gap) @ v_cl
                        cur = t
                    y_cl = v_cl / np.linalg.norm(v_cl)
                    y_ex = exact_solution_fourier(u0_vals, phys_t, nu, c_vel)

                    col = colors[idx]
                    ax.plot(x_grid, y_ex, color=col, ls="-", alpha=0.4, lw=1.5)
                    ax.plot(x_grid, y_cl, color=col, ls="--", alpha=0.6, lw=1.5)
                    ax.plot(x_grid, yq, "o", color=col, ms=5, label=f"t={t}",
                            markeredgecolor="white", markeredgewidth=0.5)
                    ax.legend(loc="upper right", fontsize=8, ncol=2)
                    plot_ph.pyplot(fig)
                    status.markdown(
                        f"**Step {t} done** — {total_valid}/{shots} valid "
                        f"({100*total_valid/shots:.1f} %)")
            except Exception as e:
                status.markdown(f"**Error at step {t}:** {e}")

            pbar.progress(idx / len(steps))

        # Final legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color="gray", lw=2, alpha=0.4, label="Exact (Fourier)"),
            Line2D([0], [0], color="gray", lw=2, ls="--", alpha=0.6, label="Classical (FD)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   ms=5, label="Quantum (QSVT)"),
        ]
        ax.legend(handles=custom_lines, loc="upper left", fontsize=10)
        plot_ph.pyplot(fig)
        plt.close(fig)
        status.markdown("**Simulation complete.**")
        pbar.progress(1.0)
