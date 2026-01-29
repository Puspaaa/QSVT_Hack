import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_split_step_sim, exact_solution_fourier, get_classical_matrix

st.set_page_config(page_title="Quantum Wind Tunnel", layout="wide")
st.title("🌊 Quantum Advection-Diffusion Simulator")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Physics Parameters")
    nu = st.slider("Viscosity (nu)", 0.005, 0.05, 0.02, step=0.001)
    c = st.slider("Advection Velocity (c)", -1.0, 1.0, 0.5)
    
    st.header("2. Quantum Hardware")
    n_qubits = st.selectbox("Number of Qubits", [4, 5, 6], index=2)
    
    st.header("3. Simulation")
    t_max = st.slider("Max Time Steps", 10, 100, 30, step=10)
    shots = st.number_input("Number of Shots", min_value=10000, max_value=500000, value=200000, step=50000)
    
    run_btn = st.button("🚀 Run Quantum Circuit", type="primary")

# --- MAIN AREA ---
if run_btn:
    with st.spinner("Compiling Quantum Circuits..."):
        # Setup Initial Condition
        u0_func = lambda x: np.exp(-100 * (x - 0.3)**2)
        steps = [0, int(t_max/4), int(t_max/2), int(3*t_max/4), t_max]
        
        # 1. Run Quantum Simulation
        results = run_split_step_sim(n_qubits, nu, c, steps, u0_func, shots=shots)
        
        # 2. Prepare Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
        
        N = 2**n_qubits
        x_grid = np.linspace(0, 1, N, endpoint=False)
        u0_vals = u0_func(x_grid)
        A_matrix, dt = get_classical_matrix(N, nu, c)
        
        # Track classical evolution incrementally (like notebook)
        v_classical = u0_vals.copy()
        current_step = 0
        
        for i, t in enumerate(steps):
            col = colors[i]
            
            # Exact Analytical Solution
            phys_time = t * dt
            y_exact = exact_solution_fourier(u0_vals, phys_time, nu, c)
            ax.plot(x_grid, y_exact, color=col, linestyle='-', alpha=0.4, lw=2, label=f"Exact (t={t})")
            
            # Classical Matrix (Incremental Evolution)
            steps_to_take = t - current_step
            if steps_to_take > 0:
                A_gap = np.linalg.matrix_power(A_matrix, steps_to_take)
                v_classical = A_gap @ v_classical
                current_step = t
            y_class = v_classical / np.linalg.norm(v_classical)
            ax.plot(x_grid, y_class, color=col, linestyle='--', alpha=0.7, linewidth=1.5, label=f"Classical Matrix")
            
            # Quantum (Dots)
            if t in results:
                xq, pq = results[t]
                yq = np.sqrt(pq)
                ax.plot(xq, yq, 'o', color=col, markersize=5, label=f"Quantum (t={t})")

        ax.set_xlabel("Position")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        # Custom legend to reduce clutter
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        st.pyplot(fig)
else:
    st.info("Adjust parameters on the left and click 'Run Quantum Circuit' to start.")