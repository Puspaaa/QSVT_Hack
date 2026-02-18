"""Slide 7b: Live Demo — 2D Advection-Diffusion PDE.

Multi-step workflow (mirrors the 1D demo):
  Step 1  Choose grid qubits → compute QSVT angles → show circuit
  Step 2  Choose physics parameters (nu, cx, cy) → show block-encoding matrix
  Step 3  Choose IC + run simulation → heatmap snapshots
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from slides.components import slide_header, key_concept

TITLE = "Demo: 2D PDE Solver"


def render():
    slide_header("Live Demo — 2D Advection-Diffusion",
                 "5-point stencil Laplacian + QFT advection on a periodic 2-D grid")

    from simulation import exact_solution_fourier_2d, get_classical_matrix_2d
    from quantum import QSVT_circuit_2d, Block_encoding_diffusion_2d, Advection_Gate_2d
    from solvers import cvx_poly_coef, Angles_Fixed

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — Grid qubits + QSVT angle computation + circuit
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Step 1 — QSVT Angle Computation (2D)")
    st.markdown(
        r"Choose $n_x$ and $n_y$ qubits for each axis "
        r"($N_x = 2^{n_x}$, $N_y = 2^{n_y}$).  "
        r"A target evolution depth is used to pre-compute angles via "
        r"Chebyshev approximation."
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        nx = st.slider("X-axis qubits ($n_x$)", 2, 6, 3, 1, key="d2d_nx")
        ny = st.slider("Y-axis qubits ($n_y$)", 2, 6, 3, 1, key="d2d_ny")
        n_total = nx + ny
        Nx, Ny = 2 ** nx, 2 ** ny
        st.caption(f"Grid: {Nx} x {Ny} = {Nx * Ny} points  ({n_total} data qubits)")

        if n_total > 12:
            st.error(f"Total qubits ({n_total}) exceeds safe limit. Reduce grid size.")
            return

        t_angles = st.slider("Polynomial target time", 5, 40, 15, 5,
                              key="d2d_t_angles")
        st.caption("Number of matrix applications the Chebyshev polynomial must approximate")
        compute_btn = st.button("Compute angles", type="primary",
                                key="d2d_compute_angles")

    # Persistent state
    if "d2d_phi" not in st.session_state:
        st.session_state.d2d_phi = None
        st.session_state.d2d_coef = None
        st.session_state.d2d_deg = None
        st.session_state.d2d_t_used = None

    if compute_btn:
        deg = int(t_angles + 8)
        if deg % 2 != 0:
            deg += 1
        with st.spinner("Solving Chebyshev optimisation + QSP angle finding …"):
            try:
                target_f = lambda x: np.exp(t_angles * (np.abs(x) - 1))
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)
                st.session_state.d2d_phi = phi_seq
                st.session_state.d2d_coef = coef
                st.session_state.d2d_deg = deg
                st.session_state.d2d_t_used = t_angles
            except Exception as exc:
                st.error(f"Angle computation failed: {exc}")

    if st.session_state.d2d_phi is not None:
        phi_seq_show = st.session_state.d2d_phi
        deg_show = st.session_state.d2d_deg
        t_used = st.session_state.d2d_t_used

        with c2:
            st.success(
                f"Computed **{len(phi_seq_show)} angles** for even-parity "
                f"Chebyshev polynomial of degree **{deg_show}** (t = {t_used})."
            )

            with st.expander("Show all QSVT angles", expanded=False):
                cols_per_row = 6
                rows = (len(phi_seq_show) + cols_per_row - 1) // cols_per_row
                for r in range(rows):
                    cs = st.columns(cols_per_row)
                    for j, col in enumerate(cs):
                        idx = r * cols_per_row + j
                        if idx < len(phi_seq_show):
                            col.metric(f"$\\phi_{{{idx}}}$",
                                       f"{phi_seq_show[idx]:.4f}")

            # Polynomial fit plot
            from numpy.polynomial.chebyshev import chebval
            x_plot = np.linspace(0, 1, 500)
            y_target = np.exp(t_used * (np.abs(x_plot) - 1))
            y_approx = chebval(x_plot, st.session_state.d2d_coef)

            fig_poly, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(7, 4), height_ratios=[3, 1],
                gridspec_kw={"hspace": 0.08})
            ax1.plot(x_plot, y_target, "b-", lw=2,
                     label=r"Target $e^{t(|x|-1)}$")
            ax1.plot(x_plot, y_approx, "r--", lw=2,
                     label=f"Chebyshev deg-{deg_show}")
            ax1.legend(fontsize=9); ax1.set_ylabel("f(x)")
            ax1.set_title(
                f"Polynomial approximation  (t = {t_used}, d = {deg_show})",
                fontsize=12, fontweight="bold")
            ax1.set_xlim(0, 1); ax1.tick_params(labelbottom=False)
            ax1.grid(True, alpha=0.3)
            err = np.abs(y_target - y_approx)
            ax2.semilogy(x_plot, err + 1e-16, "green", lw=1.5)
            ax2.set_xlabel("x"); ax2.set_ylabel("|Error|")
            ax2.set_xlim(0, 1); ax2.grid(True, alpha=0.3)
            fig_poly.tight_layout()
            st.pyplot(fig_poly, use_container_width=True)
            plt.close(fig_poly)

        # QSVT circuit
        st.markdown("#### 2D QSVT Circuit")
        try:
            nu_preview = 0.02
            qc_vis = QSVT_circuit_2d(phi_seq_show, nx, ny, nu_preview,
                                      measurement=False)
            w = max(16, len(phi_seq_show) * 1.8)
            h = max(5, n_total * 0.5 + 3)
            fig_qsvt, ax_qsvt = plt.subplots(figsize=(w, h), dpi=120)
            qc_vis.draw("mpl", fold=-1, ax=ax_qsvt, style={
                "backgroundcolor": "#FFFFFF",
                "gatefacecolor": "#85C1E2",
                "textcolor": "#000000"})
            plt.tight_layout()
            st.pyplot(fig_qsvt, use_container_width=True)
            plt.close(fig_qsvt)
        except Exception as exc:
            st.info(f"Circuit rendering skipped: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — Physics parameters → block-encoding matrix
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Step 2 — Block Encoding (2D LCU)")
    st.markdown(
        r"The 2D Laplacian uses a 5-point stencil: "
        r"$A = a_0\,I + a_{+x}\,S_x + a_{-x}\,S_x^\dagger "
        r"+ a_{+y}\,S_y + a_{-y}\,S_y^\dagger$ with 2 ancilla qubits."
    )

    p1, p2 = st.columns([1, 2])
    with p1:
        nu = st.slider("Diffusion coefficient ($\\nu$)", 0.005, 0.05, 0.02, 0.001,
                        key="d2d_nu")
        st.caption("Controls how fast the solution spreads (higher = faster diffusion)")
        cx = st.slider("X-advection velocity ($c_x$)", -1.0, 1.0, 0.3, 0.05,
                        key="d2d_cx")
        cy = st.slider("Y-advection velocity ($c_y$)", -1.0, 1.0, 0.3, 0.05,
                        key="d2d_cy")
        st.caption("Speed and direction of bulk transport along each axis")
        dx = 1.0 / Nx
        dy = 1.0 / Ny
        dt = 0.9 * min(dx ** 2, dy ** 2) / (4 * nu)
        a_center = 1.0 - 4 * dt * nu / (dx ** 2)
        st.markdown(
            f"**Derived:**  \n"
            f"$\\Delta x = {dx:.4f}$, $\\Delta y = {dy:.4f}$  \n"
            f"$\\Delta t = {dt:.6f}$  \n"
            f"$a_0 = {a_center:.6f}$"
        )

    with p2:
        A_matrix, _ = get_classical_matrix_2d(Nx, Ny, nu, cx, cy)
        dim = Nx * Ny
        fig_mat, ax_mat = plt.subplots(figsize=(5, 4))
        vmax = np.max(np.abs(A_matrix))
        im = ax_mat.imshow(A_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="equal")
        ax_mat.set_title(
            f"Time-stepping matrix  A  ({dim}x{dim})",
            fontsize=12, fontweight="bold")
        fig_mat.colorbar(im, ax=ax_mat, shrink=0.8)
        if dim <= 16:
            for i in range(dim):
                for j in range(dim):
                    v = A_matrix[i, j]
                    if abs(v) > 1e-8:
                        ax_mat.text(j, i, f"{v:.2f}", ha="center",
                                    va="center", fontsize=max(5, 10 - dim))
        fig_mat.tight_layout()
        st.pyplot(fig_mat, use_container_width=True)
        plt.close(fig_mat)

    with st.expander("Show 2D block-encoding circuit (LCU)", expanded=False):
        try:
            qc_block = Block_encoding_diffusion_2d(nx, ny, nu)
            w = max(14, n_total * 2)
            h = max(4, n_total * 0.5 + 2)
            fig_c, ax_c = plt.subplots(figsize=(w, h), dpi=120)
            qc_block.draw("mpl", fold=-1, ax=ax_c, style={
                "backgroundcolor": "#FFFFFF", "gatefacecolor": "#BB8FCE"})
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
        except Exception as exc:
            st.info(f"Circuit drawing skipped: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3 — Initial condition + simulation
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Step 3 — Initial Condition and Simulation")

    x_arr = np.linspace(0, 1, Nx, endpoint=False)
    y_arr = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")

    ic_col, prev_col = st.columns([1.2, 1])
    with ic_col:
        ic_type = st.selectbox("Function type", [
            "Gaussian Peak",
            "Double Gaussian",
            "Sine Pattern",
            "Checkerboard",
            "Ring",
            "Custom Function",
        ], key="d2d_ic")

        if ic_type == "Gaussian Peak":
            gx = st.slider("Center X", 0.1, 0.9, 0.5, 0.05, key="d2d_gx")
            gy = st.slider("Center Y", 0.1, 0.9, 0.5, 0.05, key="d2d_gy")
            gw = st.slider("Width", 5, 30, 10, 1, key="d2d_gw2")
            u0_2d = np.exp(-gw * ((X - gx) ** 2 + (Y - gy) ** 2))
        elif ic_type == "Double Gaussian":
            u0_2d = (np.exp(-8 * ((X - 0.3) ** 2 + (Y - 0.3) ** 2))
                     + np.exp(-8 * ((X - 0.7) ** 2 + (Y - 0.7) ** 2)))
        elif ic_type == "Sine Pattern":
            fx = st.slider("X frequency", 1, 4, 2, key="d2d_fx")
            fy = st.slider("Y frequency", 1, 4, 2, key="d2d_fy")
            u0_2d = np.abs(np.sin(2 * np.pi * fx * X)
                           * np.sin(2 * np.pi * fy * Y))
        elif ic_type == "Checkerboard":
            cb = st.slider("Blocks per axis", 2, 8, 4, key="d2d_cb")
            u0_2d = np.abs(np.sign(
                np.sin(np.pi * cb * X) * np.sin(np.pi * cb * Y))) + 0.1
        elif ic_type == "Ring":
            rr = st.slider("Ring radius", 0.1, 0.4, 0.25, 0.05,
                            key="d2d_rr")
            rw = st.slider("Ring width", 0.02, 0.15, 0.05, 0.01,
                            key="d2d_rw")
            dist = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
            u0_2d = np.exp(-((dist - rr) / rw) ** 2)
        else:
            st.markdown("Python expression (`X`, `Y`, `np`)")
            custom_2d = st.text_input(
                "u0(X,Y)",
                "np.exp(-10*((X-0.5)**2+(Y-0.5)**2))",
                key="d2d_custom")
            try:
                _test = eval(custom_2d, {"X": X, "Y": Y, "np": np})
                u0_2d = eval(custom_2d, {"X": X, "Y": Y, "np": np}).astype(float)
                st.success("Valid expression")
            except Exception as exc:
                st.error(f"Invalid: {exc}")
                u0_2d = np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

        st.markdown("---")
        t_max = st.slider("Evolution depth", 10, 60, 30, 10, key="d2d_tmax")
        st.caption("How many matrix-power steps to evolve — larger values show more diffusion / advection")
        n_snap = st.slider("Number of snapshots", 3, 8, 4, 1, key="d2d_snap")
        st.caption("How many evenly-spaced time frames to render")
        shots = st.number_input("Measurement shots", 10000, 200000, 100000, 10000,
                                key="d2d_shots")
        st.caption("Circuit samples — more shots reduce statistical noise in the heatmaps")

    norm_val = np.linalg.norm(u0_2d.flatten())
    u0_2d_n = u0_2d / norm_val if norm_val > 0 else u0_2d

    with prev_col:
        st.markdown("**Initial condition preview**")
        from scipy.ndimage import zoom
        u0_fine = zoom(u0_2d_n, (max(1, 128 // Nx), max(1, 128 // Ny)),
                       order=1)
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        ax_p.imshow(u0_fine, cmap="viridis", origin="lower",
                    extent=[0, 1, 0, 1], interpolation="bilinear")
        ax_p.set_title("Normalised initial state", fontsize=11,
                        fontweight="bold")
        ax_p.set_xlabel("X"); ax_p.set_ylabel("Y")
        fig_p.tight_layout()
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

    time_steps = [int(i * t_max / (n_snap - 1)) for i in range(n_snap)]

    run_btn = st.button("Run 2-D Quantum Simulation", type="primary",
                        key="d2d_run")

    if not run_btn:
        return

    # ── Execute simulation ────────────────────────────────────────────────
    A_matrix, dt_sim = get_classical_matrix_2d(Nx, Ny, nu, cx, cy)

    full_dim = 2 ** n_total
    u0_flat = u0_2d.flatten()
    u0_full = np.zeros(full_dim)
    u0_full[:len(u0_flat)] = u0_flat
    u0_full /= np.linalg.norm(u0_full)

    n_cols = min(3, len(time_steps))
    n_rows = (len(time_steps) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows))
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
            u_fine = zoom(u0_2d_n,
                          (max(1, 128 // Nx), max(1, 128 // Ny)), order=1)
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

            phys_t = t * dt_sim
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
                               (max(1, 128 // Nx), max(1, 128 // Ny)),
                               order=1)
                ax.imshow(yq_fine, cmap="viridis", origin="lower",
                          extent=[0, 1, 0, 1], interpolation="bilinear")
                ax.set_title(
                    f"t={t}  ({100 * total_valid / shots:.0f}% valid)",
                    fontsize=11, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No valid counts",
                        transform=ax.transAxes, ha="center",
                        va="center", color="red")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
        except Exception as e:
            ax.text(0.5, 0.5, str(e)[:40], transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="red")

        pbar.progress((si + 1) / len(time_steps))

    for extra in range(si + 1, n_rows * n_cols):
        axes[extra // n_cols, extra % n_cols].axis("off")

    fig.tight_layout()
    plot_ph.pyplot(fig)
    plt.close(fig)
    status.markdown("**2-D simulation complete.**")
    pbar.progress(1.0)
