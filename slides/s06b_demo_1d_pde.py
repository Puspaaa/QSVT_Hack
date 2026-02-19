"""Slide 6b: Live Demo — 1D Advection-Diffusion PDE.

Multi-step workflow
  Step 1  Choose grid + physics → derive dt → show block-encoding matrix
  Step 2  Choose k (steps per circuit) and m (number of runs) → compute angles
  Step 3  Choose initial condition + shots → Run simulation → live evolution

Each QSVT circuit implements A^k.  The output of run j is re-encoded as the
input of run j+1, so m sequential runs evolve to time T = m·k·Δt.
A resource comparison shows qubit and gate costs vs. the classical method.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from slides.components import slide_header, key_concept

TITLE = "Demo: 1D PDE Solver"


# ── helper: postselect counts → probability distribution ──────────────
def _postselect(counts, n_qubits):
    """Return (prob_dist, total_valid) from measurement counts."""
    N = 2 ** n_qubits
    prob_dist = np.zeros(N)
    total_valid = 0
    for key, cnt in counts.items():
        parts = key.split()
        s_bit = a_bits = d_bits = ""
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
    return prob_dist, total_valid


def render():
    slide_header("Live Demo — 1D Advection-Diffusion",
                 "QSVT diffusion + QFT advection on a periodic domain")

    from simulation import exact_solution_fourier, get_classical_matrix
    from quantum import (QSVT_circuit_universal,
                         Block_encoding_diffusion, Advection_Gate)
    from solvers import cvx_poly_coef, Angles_Fixed

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — Grid + physics → derive dt → block-encoding matrix
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Step 1 — Grid & Physics Setup")
    st.markdown(
        r"Set the spatial resolution and physical parameters.  "
        r"Together they determine $\Delta t$ — the physical time that "
        r"**one application** of the matrix $A$ represents."
    )

    p1, p2 = st.columns([1, 2])
    with p1:
        n_qubits = st.slider("Grid qubits ($n$)", 3, 8, 5, 1,
                              key="d1d_n")
        N = 2 ** n_qubits
        st.caption(f"Grid has $N = 2^{n_qubits} = {N}$ spatial points")
        nu = st.slider("Diffusion coefficient ($\\nu$)", 0.005, 0.05,
                        0.02, 0.001, key="d1d_nu")
        st.caption("Higher → faster spreading")
        c_vel = st.slider("Advection velocity ($c$)", -1.0, 1.0, 0.5,
                           0.05, key="d1d_c")
        st.caption("Bulk transport speed (positive = rightward)")

        dx = 1.0 / N
        dt = 0.9 * dx ** 2 / (2 * nu)
        a_val = 1 - 2 * dt * nu / dx ** 2

        st.markdown("---")
        st.markdown(
            f"**Derived from grid + $\\nu$:**  \n"
            f"$\\Delta x = 1/N = {dx:.4f}$  \n"
            f"$\\Delta t = 0.9\\,\\Delta x^2 / (2\\nu) = {dt:.6f}$  \n"
            f"$a_0 = {a_val:.6f}$"
        )
        st.info(
            f"**One matrix application** $A|\\psi\\rangle$ advances the "
            f"solution by $\\Delta t = {dt:.6f}$ time units."
        )

    with p2:
        A_matrix, _ = get_classical_matrix(N, nu, c_vel)
        fig_mat, ax_mat = plt.subplots(figsize=(5, 4))
        vmax = np.max(np.abs(A_matrix))
        im = ax_mat.imshow(A_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="equal")
        ax_mat.set_title(
            f"Time-stepping matrix  $A$  ({N}×{N})",
            fontsize=12, fontweight="bold")
        fig_mat.colorbar(im, ax=ax_mat, shrink=0.8)
        if N <= 16:
            for i in range(N):
                for j in range(N):
                    v = A_matrix[i, j]
                    if abs(v) > 1e-8:
                        ax_mat.text(j, i, f"{v:.2f}", ha="center",
                                    va="center",
                                    fontsize=max(6, 11 - N))
        fig_mat.tight_layout()
        st.pyplot(fig_mat, use_container_width=True)
        plt.close(fig_mat)

        st.caption(
            r"$A = a_0\,I + a_+\,S + a_-\,S^\dagger$ — the LCU "
            r"decomposition uses just 2 ancilla qubits."
        )

    with st.expander("Show block-encoding circuit (LCU)", expanded=False):
        try:
            qc_block = Block_encoding_diffusion(n_qubits, nu)
            w = max(14, n_qubits * 2)
            h = max(3, n_qubits * 0.5 + 2)
            fig_c, ax_c = plt.subplots(figsize=(w, h), dpi=120)
            qc_block.draw("mpl", fold=-1, ax=ax_c, style={
                "backgroundcolor": "#FFFFFF",
                "gatefacecolor": "#BB8FCE"})
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
        except Exception as exc:
            st.info(f"Circuit drawing skipped: {exc}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — Steps per circuit (k) + number of runs (m) → QSVT angles
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Step 2 — Circuit Design & QSVT Angles")

    st.markdown(
        r"""
**How the quantum simulation works:**
each QSVT circuit implements $A^k$ — it applies $k$ time-steps in a
*single* quantum circuit.  After measurement and post-selection we
obtain the state at time $T_{\text{step}} = k \cdot \Delta t$.

To see the solution evolve, we **chain** $m$ runs: the measured output
of run $j$ is re-encoded as the input of run $j{+}1$.  After $m$ runs
the solution has reached $T = m \cdot k \cdot \Delta t$.

> **Key difference from classical:** classically you can inspect *every*
> intermediate step at no extra cost.  Quantumly, you only get the
> final state of each circuit run — intermediate states are
> inaccessible inside the circuit.
"""
    )

    c1, c2 = st.columns([1, 3])
    dt_phys = dt
    with c1:
        k_steps = st.slider(
            "Steps per circuit ($k$)", 1, 60, 10, 1, key="d1d_k",
            help="Number of matrix applications implemented by one "
                 "QSVT circuit.  Larger k → higher polynomial degree "
                 "→ deeper circuit."
        )
        st.caption(
            f"One circuit run advances by "
            f"$k \\cdot \\Delta t = {k_steps} \\times {dt_phys:.6f}"
            f" = {k_steps * dt_phys:.6f}$ time units."
        )

        m_runs = st.slider(
            "Number of circuit runs ($m$)", 1, 12, 5, 1, key="d1d_m",
            help="Each run feeds the measured output of the previous "
                 "run as its initial state.  m runs → m snapshots."
        )
        T_total = m_runs * k_steps * dt_phys
        st.caption(
            f"Total physical time: "
            f"$T = m \\cdot k \\cdot \\Delta t = {m_runs} \\times "
            f"{k_steps} \\times {dt_phys:.6f} = {T_total:.6f}$"
        )

        st.markdown("---")
        compute_btn = st.button("Compute QSVT angles", type="primary",
                                key="d1d_compute_angles")

    # Persistent state for angles
    if "d1d_phi" not in st.session_state:
        st.session_state.d1d_phi = None
        st.session_state.d1d_coef = None
        st.session_state.d1d_deg = None
        st.session_state.d1d_k_used = None

    if compute_btn:
        deg = int(k_steps + 8)
        if deg % 2 != 0:
            deg += 1
        with st.spinner(
                "Solving Chebyshev optimisation + QSP angle finding …"):
            try:
                target_f = lambda x: np.exp(k_steps * (np.abs(x) - 1))
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)
                st.session_state.d1d_phi = phi_seq
                st.session_state.d1d_coef = coef
                st.session_state.d1d_deg = deg
                st.session_state.d1d_k_used = k_steps
            except Exception as exc:
                st.error(f"Angle computation failed: {exc}")

    if st.session_state.d1d_phi is not None:
        phi_seq_show = st.session_state.d1d_phi
        deg_show = st.session_state.d1d_deg
        k_used = st.session_state.d1d_k_used

        with c2:
            st.success(
                f"Computed **{len(phi_seq_show)} QSVT angles** — "
                f"even polynomial of degree **{deg_show}** "
                f"approximating $A^{{{k_used}}}$ "
                f"(one run = {k_used * dt_phys:.6f} time units)."
            )

            # --- angle table ------------------------------------------------
            with st.expander("Show all QSVT angles", expanded=False):
                cols_per_row = 6
                rows = ((len(phi_seq_show) + cols_per_row - 1)
                        // cols_per_row)
                for r in range(rows):
                    cs = st.columns(cols_per_row)
                    for j, col in enumerate(cs):
                        idx = r * cols_per_row + j
                        if idx < len(phi_seq_show):
                            col.metric(f"$\\phi_{{{idx}}}$",
                                       f"{phi_seq_show[idx]:.4f}")

            # --- polynomial fit plot -----------------------------------------
            from numpy.polynomial.chebyshev import chebval
            x_plot = np.linspace(0, 1, 500)
            y_target = np.exp(k_used * (np.abs(x_plot) - 1))
            y_approx = chebval(x_plot, st.session_state.d1d_coef)

            fig_poly, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(7, 4), height_ratios=[3, 1],
                gridspec_kw={"hspace": 0.08})
            ax1.plot(x_plot, y_target, "b-", lw=2,
                     label=r"Target $e^{k(|x|-1)}$")
            ax1.plot(x_plot, y_approx, "r--", lw=2,
                     label=f"Chebyshev deg-{deg_show}")
            ax1.legend(fontsize=9)
            ax1.set_ylabel("f(x)")
            ax1.set_title(
                f"Polynomial approximation  (k = {k_used} steps, "
                f"d = {deg_show})",
                fontsize=12, fontweight="bold")
            ax1.set_xlim(0, 1)
            ax1.tick_params(labelbottom=False)
            ax1.grid(True, alpha=0.3)
            err = np.abs(y_target - y_approx)
            ax2.semilogy(x_plot, err + 1e-16, "green", lw=1.5)
            ax2.set_xlabel("x")
            ax2.set_ylabel("|Error|")
            ax2.set_xlim(0, 1)
            ax2.grid(True, alpha=0.3)
            fig_poly.tight_layout()
            st.pyplot(fig_poly, use_container_width=True)
            plt.close(fig_poly)

        # --- QSVT circuit with angle labels ---------------------------------
        st.markdown("#### QSVT Circuit (with labelled angles)")
        try:
            qc_vis = QSVT_circuit_universal(
                phi_seq_show, n_qubits, nu, measurement=False)
            w = max(16, len(phi_seq_show) * 1.8)
            h = max(4, n_qubits * 0.6 + 2)
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
    #  STEP 3 — Initial condition + simulation
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### Step 3 — Initial Condition & Run")

    ic_col, prev_col = st.columns([1.2, 1])

    with ic_col:
        ic_type = st.selectbox("Function type", [
            "Gaussian Peak",
            "Double Gaussian",
            "Sine Wave",
            "Square Wave",
            "Triangle Wave",
            "Custom Function",
        ], key="d1d_ic")

        if ic_type == "Gaussian Peak":
            center = st.slider("Peak center", 0.0, 1.0, 0.3, 0.05,
                               key="d1d_gc")
            width = st.slider("Peak width", 10, 200, 100, 10,
                              key="d1d_gw")
            u0_func = lambda x, _c=center, _w=width: np.exp(
                -_w * (x - _c) ** 2)
        elif ic_type == "Double Gaussian":
            c1x = st.slider("Peak 1 center", 0.0, 0.5, 0.25, 0.05,
                             key="d1d_dg1")
            c2x = st.slider("Peak 2 center", 0.5, 1.0, 0.75, 0.05,
                             key="d1d_dg2")
            width = st.slider("Peak width", 10, 200, 80, 10,
                              key="d1d_dgw")
            u0_func = lambda x, _a=c1x, _b=c2x, _w=width: (
                np.exp(-_w * (x - _a) ** 2)
                + np.exp(-_w * (x - _b) ** 2))
        elif ic_type == "Sine Wave":
            freq = st.slider("Frequency", 1, 5, 2, key="d1d_sf")
            u0_func = lambda x, _f=freq: np.abs(
                np.sin(2 * np.pi * _f * x))
        elif ic_type == "Square Wave":
            sq_c = st.slider("Pulse center", 0.0, 1.0, 0.5, 0.05,
                              key="d1d_sqc")
            sq_w = st.slider("Pulse width", 0.1, 0.5, 0.2, 0.05,
                              key="d1d_sqw")
            u0_func = lambda x, _c=sq_c, _w=sq_w: np.where(
                np.abs((x - _c + 0.5) % 1.0 - 0.5) < _w / 2, 1.0, 0.1)
        elif ic_type == "Triangle Wave":
            tri_c = st.slider("Peak position", 0.0, 1.0, 0.5, 0.05,
                               key="d1d_trc")
            tri_w = st.slider("Base width", 0.1, 0.5, 0.3, 0.05,
                               key="d1d_trw")
            def _triangle(x, _c=tri_c, _w=tri_w):
                dist = np.abs((x - _c + 0.5) % 1.0 - 0.5)
                return np.maximum(0, 1 - dist / (_w / 2)) + 0.1
            u0_func = _triangle
        else:  # Custom Function
            st.markdown(
                "Enter a Python expression (variable `x`, numpy as `np`)")
            custom_expr = st.text_input(
                "u0(x)", "np.exp(-100*(x-0.3)**2)", key="d1d_custom")
            try:
                _test = eval(custom_expr,
                             {"x": np.linspace(0, 1, 10), "np": np})
                u0_func = lambda x, _e=custom_expr: eval(
                    _e, {"x": x, "np": np})
                st.success("Valid expression")
            except Exception as exc:
                st.error(f"Invalid: {exc}")
                u0_func = lambda x: np.exp(-100 * (x - 0.3) ** 2)

        st.markdown("---")
        shots = st.number_input("Measurement shots", 10000, 500000,
                                100000, 50000, key="d1d_shots")
        st.caption("Samples per circuit run — more shots = less noise")

    x_grid = np.linspace(0, 1, N, endpoint=False)
    u0_vals = u0_func(x_grid)

    with prev_col:
        st.markdown("**Initial condition preview**")
        y_init = u0_vals / np.linalg.norm(u0_vals)
        fig_p, ax_p = plt.subplots(figsize=(5, 3.5))
        ax_p.plot(x_grid, y_init, "b-", lw=2)
        ax_p.fill_between(x_grid, y_init, alpha=0.2)
        ax_p.set_xlabel("x")
        ax_p.set_ylabel("u(x, 0)")
        ax_p.set_title("Normalised initial state", fontsize=11,
                        fontweight="bold")
        ax_p.grid(True, alpha=0.3)
        fig_p.tight_layout()
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

    # Show the run plan
    snap_md = "  →  ".join(
        [f"Run {j}: $T = {j * k_steps * dt_phys:.4f}$"
         for j in range(1, m_runs + 1)]
    )
    st.caption(f"**Plan:** {snap_md}")

    run_btn = st.button("Run 1-D Quantum Simulation", type="primary",
                        key="d1d_run")

    if not run_btn:
        return

    # ── Execute simulation ────────────────────────────────────────────────
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    A_mat, dt_sim = get_classical_matrix(N, nu, c_vel)

    fig, ax = plt.subplots(figsize=(12, 6))
    n_colors = m_runs + 1
    colors = plt.cm.viridis(np.linspace(0, 1, n_colors))

    y0 = u0_vals / np.linalg.norm(u0_vals)
    ax.plot(x_grid, y0, color=colors[0], lw=2.5,
            label="$T = 0$")
    ax.fill_between(x_grid, y0, alpha=0.15, color=colors[0])
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_title(
        f"1-D Advection-Diffusion  ($n = {n_qubits}$, "
        f"$\\nu = {nu}$, $c = {c_vel}$)",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    method_handles = [
        Line2D([0], [0], color="gray", lw=1.5, alpha=0.4,
               label="Exact (Fourier)"),
        Line2D([0], [0], color="gray", lw=1.5, ls="--", alpha=0.6,
               label="Classical (FD)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               ms=5, label="Quantum (QSVT)"),
    ]

    plot_ph = st.empty()
    plot_ph.pyplot(fig)
    status = st.empty()
    pbar = st.progress(0)

    # --- chained simulation loop ------------------------------------------
    # Quantum state that gets re-encoded each run
    q_state = u0_vals / np.linalg.norm(u0_vals)
    # Classical reference (accumulated matrix power)
    v_cl = u0_vals.copy()

    time_handles = []
    total_q_gates = 0     # accumulated gate count across all runs
    total_q_depth = 0     # accumulated circuit depth
    backend = AerSimulator()
    phi_seq = st.session_state.d1d_phi

    for j in range(1, m_runs + 1):
        total_matrix_steps = j * k_steps
        phys_t = total_matrix_steps * dt_phys
        status.markdown(
            f"**Circuit run {j}/{m_runs}:  "
            f"$T = {phys_t:.4f}$ …**"
        )

        try:
            # Build circuit with CURRENT quantum state as input
            qc = QSVT_circuit_universal(
                phi_seq, n_qubits, nu,
                init_state=q_state, measurement=False)

            # Append advection for this run's physical time slice
            adv_t = k_steps * dt_phys
            qc.append(Advection_Gate(n_qubits, c_vel, adv_t),
                       qc.qregs[2])

            qc.measure(qc.qregs[0], qc.cregs[0])
            qc.measure(qc.qregs[1], qc.cregs[1])
            qc.measure(qc.qregs[2], qc.cregs[2])

            tqc = transpile(qc, backend, optimization_level=0)
            total_q_gates += tqc.size()
            total_q_depth += tqc.depth()

            counts = backend.run(tqc, shots=shots).result().get_counts()

            prob_dist, total_valid = _postselect(counts, n_qubits)

            if total_valid > 0:
                yq = np.sqrt(prob_dist)
                # Re-encode measured output as next run's initial state
                q_state = yq / np.linalg.norm(yq) if np.linalg.norm(yq) > 0 else q_state

                # Classical comparison: advance to the same total step
                v_cl = np.linalg.matrix_power(A_mat, k_steps) @ v_cl
                y_cl = v_cl / np.linalg.norm(v_cl)
                y_ex = exact_solution_fourier(u0_vals, phys_t, nu, c_vel)

                col = colors[j]
                ax.plot(x_grid, y_ex, color=col, ls="-", alpha=0.4,
                        lw=1.5)
                ax.plot(x_grid, y_cl, color=col, ls="--", alpha=0.6,
                        lw=1.5)
                ax.plot(x_grid, yq, "o", color=col, ms=5,
                        markeredgecolor="white", markeredgewidth=0.5)

                time_handles.append(
                    Line2D([0], [0], marker="o", color=col,
                           markerfacecolor=col, ms=6, lw=0,
                           label=(f"Run {j}: $T = {phys_t:.4f}$")))

                ax.legend(
                    handles=method_handles + time_handles,
                    loc="upper right", fontsize=8, ncol=2,
                    handlelength=1.5)
                plot_ph.pyplot(fig)
                status.markdown(
                    f"**Run {j} done** — "
                    f"T = {phys_t:.4f} — "
                    f"{total_valid}/{shots} valid "
                    f"({100 * total_valid / shots:.1f} %)")
            else:
                status.warning(
                    f"Run {j}: no valid post-selected counts.")
        except Exception as e:
            status.markdown(f"**Error in run {j}:** {e}")
        pbar.progress(j / m_runs)

    # Final legend
    ax.legend(
        handles=method_handles + time_handles,
        loc="upper right", fontsize=9, ncol=2, handlelength=1.5)
    plot_ph.pyplot(fig)
    plt.close(fig)
    status.markdown("**Simulation complete.**")
    pbar.progress(1.0)

    # ══════════════════════════════════════════════════════════════════════
    #  RESOURCE COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Resource Comparison: Quantum vs Classical")

    total_matrix_steps_all = m_runs * k_steps
    q_qubits = n_qubits + 3   # data + signal + 2 ancilla
    cl_ops_per_step = N * 3    # tridiagonal: ~3N multiplies per mat-vec
    cl_total_ops = cl_ops_per_step * total_matrix_steps_all

    col_q, col_c = st.columns(2)
    with col_q:
        st.markdown("#### Quantum (QSVT)")
        st.markdown(
            f"- **Qubits:** {q_qubits} "
            f"({n_qubits} data + 1 signal + 2 ancilla)\n"
            f"- **Circuit runs:** {m_runs}\n"
            f"- **Gates per run:** ~{total_q_gates // m_runs:,}\n"
            f"- **Total gates:** {total_q_gates:,}\n"
            f"- **Total depth:** {total_q_depth:,}\n"
            f"- **Shots per run:** {shots:,}\n"
            f"- **Total shots:** {m_runs * shots:,}"
        )
    with col_c:
        st.markdown("#### Classical (Finite Difference)")
        st.markdown(
            f"- **Memory:** {N}×{N} matrix = "
            f"{N * N:,} entries\n"
            f"- **Total time-steps:** {total_matrix_steps_all}\n"
            f"- **Ops per step:** ~{cl_ops_per_step:,} "
            f"(sparse mat-vec)\n"
            f"- **Total operations:** ~{cl_total_ops:,}\n"
            f"- **Inspectable steps:** all {total_matrix_steps_all} "
            f"(no measurement needed)"
        )

    st.info(
        f"**Key trade-off:** the quantum circuit uses only "
        f"**{q_qubits} qubits** (logarithmic in $N = {N}$) but "
        f"can only observe the state at the end of each circuit run "
        f"({m_runs} snapshots).  The classical method stores the full "
        f"$N$-vector at every step."
    )
