"""Slide 8b: Live Demo — Quantum Numerical Integration (3 methods)."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial.chebyshev import chebval

from slides.components import slide_header

TITLE = "Demo: Quantum Integration"


def render():
    slide_header("Live Demo — Quantum Integration",
                 "Three methods: Compute-Uncompute, Arithmetic Comparator, QSVT Parity")

    from measurements import (
        run_overlap_integral,
        run_qsvt_integral_arbitrary,
        run_arithmetic_integral,
        get_function_data,
        get_boxcar_targets,
    )
    from solvers import robust_poly_coef, Angles_Fixed

    # ── Controls ──────────────────────────────────────────────────────────
    col_fn, col_meth = st.columns(2)

    with col_fn:
        st.markdown("#### Function to integrate")
        func_choice = st.selectbox(
            "f(x)",
            ["sin", "gaussian", "linear", "quadratic", "cosine"],
            key="dint_func",
        )
        n_qubits = st.slider("Qubits (n)", 4, 10, 6, key="dint_n")
        shots = st.select_slider(
            "Shots",
            options=[1000, 5000, 10000, 20000, 50000],
            value=10000,
            key="dint_shots",
        )
        st.caption(f"Grid: {2**n_qubits} points")

    with col_meth:
        st.markdown("#### Method and interval")
        method = st.radio(
            "Integration method",
            [
                "Compute-Uncompute (special intervals)",
                "Arithmetic / Comparison (any interval)",
                "QSVT Parity Decomposition (any interval)",
            ],
            key="dint_method",
        )

        if "Compute-Uncompute" in method:
            interval_choice = st.radio(
                "Select domain D",
                ["Left Half [0, 0.5]", "Middle Half [0.25, 0.75]"],
                key="dint_cu_int",
                horizontal=True,
            )
            if "Left" in interval_choice:
                interval_id = "left_half"
                a_val, b_val = 0.0, 0.5
            else:
                interval_id = "middle_half"
                a_val, b_val = 0.25, 0.75
        elif "Arithmetic" in method:
            a_val, b_val = st.slider(
                "Interval [a, b]", 0.0, 1.0, (0.25, 0.75), 0.01,
                key="dint_arith_ab",
            )
            interval_id = None
        else:
            a_val, b_val = st.slider(
                "Interval [a, b]", 0.0, 1.0, (0.3, 0.7), 0.01,
                key="dint_qsvt_ab",
            )
            interval_id = None

    # ── Function preview ──────────────────────────────────────────────────
    N = 2 ** n_qubits
    x_plot = np.linspace(0, 1, N, endpoint=False)
    y_plot = _get_y(func_choice, x_plot)

    fig_prev, ax_prev = plt.subplots(figsize=(10, 3))
    ax_prev.plot(x_plot, y_plot, "b-", lw=2, label="f(x)")
    mask = (x_plot >= a_val) & (x_plot < b_val)
    ax_prev.fill_between(x_plot, 0, y_plot, where=mask, alpha=0.3, color="green",
                         label=f"Domain [{a_val:.2f}, {b_val:.2f}]")
    ax_prev.axvline(a_val, color="red", ls="--", lw=1.5, alpha=0.7)
    ax_prev.axvline(b_val, color="red", ls="--", lw=1.5, alpha=0.7)
    exact_int = np.sum(y_plot[mask]) / N
    ax_prev.set_title(
        f"f(x) on [{a_val:.2f}, {b_val:.2f}]  —  Exact integral = {exact_int:.5f}",
        fontsize=11, fontweight="bold",
    )
    ax_prev.legend(loc="upper right")
    ax_prev.grid(True, alpha=0.3)
    ax_prev.set_xlim(0, 1)
    fig_prev.tight_layout()
    st.pyplot(fig_prev, use_container_width=True)
    plt.close(fig_prev)

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────
    run_btn = st.button("Run Quantum Integration", type="primary", key="dint_run")

    if not run_btn:
        return

    with st.spinner("Running quantum integration …"):
        if "Compute-Uncompute" in method:
            res = run_overlap_integral(n_qubits, func_choice, interval_id, shots)
            _show_cu_results(res, n_qubits)

        elif "Arithmetic" in method:
            res = run_arithmetic_integral(n_qubits, func_choice, a_val, b_val, shots)
            _show_arith_results(res, n_qubits, a_val, b_val, func_choice)

        else:  # QSVT
            res = run_qsvt_integral_arbitrary(n_qubits, func_choice, a_val, b_val, shots)
            _show_qsvt_results(res, n_qubits, a_val, b_val, func_choice)


# ── Helper: function values ───────────────────────────────────────────────

def _get_y(name, x):
    if name == "sin":
        return np.sin(2 * np.pi * x) + 2.0
    if name == "gaussian":
        return np.exp(-20 * (x - 0.5) ** 2)
    if name == "linear":
        return 2 * x + 0.5
    if name == "quadratic":
        return 4 * x * (1 - x) + 0.5
    if name == "cosine":
        return np.cos(2 * np.pi * x) + 1.5
    return np.ones_like(x)


# ── Result renderers ──────────────────────────────────────────────────────

def _show_cu_results(res, n_qubits):
    st.success("Compute-Uncompute complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Exact", f"{res['integral_exact']:.5f}")
    c2.metric("Quantum", f"{res['integral_est']:.5f}")
    rel = abs(res["error"] / (res["integral_exact"] + 1e-12)) * 100
    c3.metric("Rel. error", f"{rel:.2f} %")


def _show_arith_results(res, n_qubits, a_val, b_val, func_choice):
    st.success("Arithmetic / Comparison complete")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Exact [{a_val:.2f}, {b_val:.2f}]", f"{res['integral_exact']:.5f}")
    c2.metric("Quantum", f"{res['integral_est']:.5f}")
    rel = abs(res["error"] / (res["integral_exact"] + 1e-12)) * 100
    c3.metric("Rel. error", f"{rel:.1f} %")

    # State-marking bar chart
    N = 2 ** n_qubits
    x_grid = np.linspace(0, 1, N, endpoint=False)
    y_vals = _get_y(func_choice, x_grid)
    amp = y_vals / np.linalg.norm(y_vals)
    a_int = int(np.floor(a_val * N))
    b_int = min(int(np.floor(b_val * N)), N - 1)
    colors = ["#2ecc71" if a_int <= j <= b_int else "#bdc3c7" for j in range(N)]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(N), amp, color=colors, edgecolor="black", alpha=0.8)
    ax.axvline(a_int - 0.5, color="red", ls="--", lw=2, label=f"a = {a_val}")
    ax.axvline(b_int + 0.5, color="blue", ls="--", lw=2, label=f"b = {b_val}")
    ax.set_xlabel("Basis state index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Marked states (green) vs unmarked (grey)", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _show_qsvt_results(res, n_qubits, a_val, b_val, func_choice):
    from measurements import get_boxcar_targets
    from solvers import robust_poly_coef

    st.success("QSVT Parity Decomposition complete")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Exact [{a_val:.2f}, {b_val:.2f}]", f"{res['integral_exact']:.5f}")
    c2.metric("QSVT estimate", f"{res['integral_est']:.5f}")
    c3.metric("Poly degrees", f"Even: {res['deg_even']} / Odd: {res['deg_odd']}")

    # Boxcar polynomial visualization
    N = 2 ** n_qubits
    x_fine = np.linspace(0, 1, 500)
    lam_fine = np.cos(np.pi * x_fine)

    t_even, t_odd, scale = get_boxcar_targets(a_val, b_val)
    coef_even = robust_poly_coef(t_even, [-1, 1], res["deg_even"], epsil=2e-3)
    coef_odd = robust_poly_coef(t_odd, [-1, 1], res["deg_odd"], epsil=2e-3)

    if coef_even is not None and coef_odd is not None:
        poly_total = chebval(lam_fine, coef_even) + chebval(lam_fine, coef_odd)
        ideal = np.where((x_fine >= a_val) & (x_fine <= b_val), scale, 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Position space
        ax1.fill_between(x_fine, ideal, alpha=0.3, color="green", label="Target")
        ax1.plot(x_fine, poly_total, "k-", lw=2, label="QSVT polynomial")
        ax1.axvline(a_val, color="red", ls="--", lw=1.5)
        ax1.axvline(b_val, color="blue", ls="--", lw=1.5)
        ax1.set_xlabel("x"); ax1.set_ylabel("Window")
        ax1.set_title("Position space", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.2, 1.3)

        # Eigenvalue space
        lam_dense = np.linspace(-1, 1, 500)
        l_a, l_b = np.cos(np.pi * a_val), np.cos(np.pi * b_val)
        l_min, l_max = min(l_a, l_b), max(l_a, l_b)
        mask_l = (lam_dense >= l_min) & (lam_dense <= l_max)
        ax2.fill_between(lam_dense, 0, scale, where=mask_l, alpha=0.3, color="green")
        poly_l = chebval(lam_dense, coef_even) + chebval(lam_dense, coef_odd)
        ax2.plot(lam_dense, poly_l, "k-", lw=2, label="Polynomial")
        ax2.axvline(l_min, color="red", ls="--"); ax2.axvline(l_max, color="blue", ls="--")
        ax2.set_xlabel("lambda"); ax2.set_ylabel("P(lambda)")
        ax2.set_title("Eigenvalue space", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-1.1, 1.1)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
