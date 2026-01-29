import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_split_step_sim, exact_solution_fourier, get_classical_matrix
from quantum import QSVT_circuit_universal, Block_encoding_diffusion, Advection_Gate
from solvers import cvx_poly_coef, Angles_Fixed

st.set_page_config(page_title="QSVT PDE Solver - 1D", layout="wide")
st.title("1D Advection-Diffusion: Quantum Singular Value Transformation")

st.markdown(r"""
### Theory Snapshot (1D)

$$\partial_t u = \nu \partial_x^2 u - c\,\partial_x u,\quad x\in[0,1)$$

**Operator flow:**
$$u_0 \;\xrightarrow{\text{Block-encode }A}\; \text{QSVT }P(A) \;\xrightarrow{\text{QFT advection}}\; u(t)$$

- Grid: $N=2^n$ points in $n$ qubits
- Diffusion: sparse Laplacian via LCU block-encoding
- Advection: phase shifts in Fourier space
""")

if 'angles_computed' not in st.session_state:
    st.session_state.angles_computed = False
if 'phi_sequences' not in st.session_state:
    st.session_state.phi_sequences = {}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Quantum Hardware")
    n_qubits = st.slider("Number of Qubits", min_value=3, max_value=8, value=6, step=1)
    st.caption(f"Grid resolution: {2**n_qubits} points")
    st.header("Physics Parameters")
    nu = st.slider("Viscosity (ν)", 0.005, 0.05, 0.02, step=0.001)
    c = st.slider("Advection Velocity (c)", -1.0, 1.0, 0.5, step=0.05)
    
    st.header("Time Evolution")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t_max = st.slider("Max Time Steps", 10, 100, 30, step=10)
    with col_t2:
        n_timesteps = st.slider("Number of Time Steps", 5, 20, 7, step=1)
    
    time_steps_display = [int(i * t_max / (n_timesteps - 1)) for i in range(n_timesteps)]
    st.info(f"Visualizing {len(time_steps_display)} time steps: {time_steps_display}")

# --- MAIN CONTENT ---

st.header("Step 1: The Advection-Diffusion Equation")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    1D advection–diffusion with periodic boundary conditions:
    """)
    st.latex(r"\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2} - c \frac{\partial u}{\partial x}")
    st.markdown("""
    - $u(x,t)$: field value
    - $\nu$: diffusion (spreading)
    - $c$: advection (shift)
    - $x \in [0,1)$ periodic
    """)

with col2:
    st.info(f"""
    **Current Parameters:**
    - ν = {nu}
    - c = {c}
    - Grid points: {2**n_qubits}
    - dx = {1/(2**n_qubits):.4f}
    - dt ≈ {0.9*(1/(2**n_qubits))**2/(2*nu):.5f}
    """)

# Section 2: Block Encoding Circuit
st.header("Step 2: Block-Encoded Unitary for Diffusion")
st.markdown("""
The diffusion operator is encoded using **Linear Combination of Unitaries (LCU)** with 2 ancilla qubits:
""")

col1, col2 = st.columns([1, 1])
with col1:
    st.latex(r"A = a_0 I + a_+ S + a_- S^\dagger")
    st.markdown("""
    - $S$ = Cyclic shift operator (implemented via QFT)
    - $a_0, a_+, a_-$ = Coefficients from finite difference stencil
    - Ancilla postselection on $|00\\rangle$ projects onto $A$
    """)

with col2:
    if st.button("Visualize Block Encoding Circuit"):
        with st.spinner("Generating circuit diagram..."):
            qc_block = Block_encoding_diffusion(n_qubits, nu)
            
            # Calculate proper figure size based on circuit depth
            width = max(14, n_qubits * 2)
            height = max(4, n_qubits * 0.6 + 2)
            
            fig, ax = plt.subplots(figsize=(width, height), dpi=150)
            qc_block.draw('mpl', 
                         style={'backgroundcolor': '#FFFFFF',
                                'textcolor': '#000000',
                                'gatetextcolor': '#000000',
                                'subtextcolor': '#000000',
                                'linecolor': '#000000',
                                'creglinecolor': '#778899',
                                'gatefacecolor': '#BB8FCE',
                                'barrierfacecolor': '#CCCCCC'},
                         fold=-1, 
                         ax=ax,
                         scale=1.0)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# Section 3: QSVT Circuit
st.header("Step 3: QSVT Circuit Structure")
st.markdown("""
**Quantum Singular Value Transformation** applies a polynomial transformation $P(A)$ to the block-encoded operator:
""")

col1, col2 = st.columns([1, 1])
with col1:
    st.latex(r"P(A) = e^{(A-I) \cdot t} \approx \sum_{k=0}^d c_k T_k(A)")
    st.markdown("""
    - Chebyshev polynomial approximation of matrix exponential
    - Degree $d$ increases with time step $t$
    - Requires $d+1$ rotation angles $\\{\\phi_0, \\phi_1, ..., \\phi_d\\}$
    """)

with col2:
    if st.button("Visualize QSVT Circuit"):
        with st.spinner("Generating QSVT diagram..."):
            # Use small degree for visualization clarity
            deg_viz = 7
            dummy_angles = np.ones(deg_viz + 1) * np.pi/4
            
            qc_qsvt = QSVT_circuit_universal(dummy_angles, n_qubits, nu, measurement=False)
            
            # Calculate proper figure size
            width = max(16, n_qubits * 3)
            height = max(6, n_qubits * 0.8 + 3)
            
            fig, ax = plt.subplots(figsize=(width, height), dpi=150)
            qc_qsvt.draw('mpl',
                        style={'backgroundcolor': '#FFFFFF',
                               'textcolor': '#000000',
                               'gatetextcolor': '#000000',
                               'subtextcolor': '#000000',
                               'linecolor': '#000000',
                               'creglinecolor': '#778899',
                               'gatefacecolor': '#85C1E2',
                               'barrierfacecolor': '#CCCCCC'},
                        fold=-1,
                        ax=ax,
                        scale=0.9)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            st.info(f"Showing QSVT with {len(dummy_angles)} angles on {n_qubits} qubits. Actual circuits will have more angles for larger time steps.")

# Section 4: Angle Computation
st.header("Step 4: Compute QSVT Angles")
st.markdown("""
The rotation angles are computed by solving a constrained optimization problem to approximate the target polynomial.
""")

calc_angles_btn = st.button("Calculate Angles for Selected Time Steps", type="primary")

if calc_angles_btn:
    with st.spinner("Computing QSVT angles..."):
        st.session_state.phi_sequences = {}
        progress_bar = st.progress(0)
        
        for idx, t in enumerate(time_steps_display):
            if t == 0:
                continue
            
            # [FIX] FORCE EVEN PARITY
            deg = int(t + 8)
            if deg % 2 != 0:
                deg += 1
            
            try:
                # [FIX] SYMMETRIC TARGET
                target_f = lambda x: np.exp(t * (np.abs(x) - 1))
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)
                st.session_state.phi_sequences[t] = phi_seq
            except Exception as e:
                st.error(f"Failed to compute angles for t={t}: {e}")
            
            progress_bar.progress((idx + 1) / len(time_steps_display))
        
        st.session_state.angles_computed = True
        st.success(f"✓ Successfully computed angles for {len(st.session_state.phi_sequences)} time steps!")

if st.session_state.angles_computed:
    st.markdown("**Angle Sequence Information:**")
    angle_data = []
    for t, phi in st.session_state.phi_sequences.items():
        angle_data.append({
            "Time Step": t,
            "Polynomial Degree": len(phi) - 1,
            "Circuit Depth": f"~{(len(phi)-1)*2} gates",
            "Angles": f"[{phi[0]:.3f}, ..., {phi[-1]:.3f}]"
        })
    st.dataframe(angle_data, use_container_width=True)

# Section 5: Initial Condition & Simulation
if st.session_state.angles_computed:
    st.header("Step 5: Choose Initial Condition and Simulate")
    
    # Display time-color legend
    st.markdown("### Time Step Color Legend")
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps_display)))
    legend_cols = st.columns(min(7, len(time_steps_display)))
    for idx, (t, col) in enumerate(zip(time_steps_display, colors)):
        with legend_cols[idx % len(legend_cols)]:
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(col[0]*255), int(col[1]*255), int(col[2]*255))
            st.markdown(f"<span style='color:{color_hex}'>■</span> t = {t}", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ic_type = st.selectbox("Function Type", [
            "Gaussian Peak",
            "Double Gaussian",
            "Sine Wave",
            "Square Wave",
            "Triangle Wave",
            "Custom Function"
        ])
        
        if ic_type == "Gaussian Peak":
            center = st.slider("Peak Center", 0.0, 1.0, 0.3, step=0.05)
            width = st.slider("Peak Width", 10, 200, 100, step=10)
            u0_func = lambda x: np.exp(-width * (x - center)**2)
        
        elif ic_type == "Double Gaussian":
            center1 = st.slider("Peak 1 Center", 0.0, 0.5, 0.25, step=0.05)
            center2 = st.slider("Peak 2 Center", 0.5, 1.0, 0.75, step=0.05)
            width = st.slider("Peak Width", 10, 200, 80, step=10)
            u0_func = lambda x: np.exp(-width * (x - center1)**2) + np.exp(-width * (x - center2)**2)
        
        elif ic_type == "Sine Wave":
            frequency = st.slider("Frequency", 1, 5, 2)
            phase = st.slider("Phase Shift", 0.0, 1.0, 0.0, step=0.1)
            u0_func = lambda x: np.abs(np.sin(2 * np.pi * frequency * (x + phase)))
        
        elif ic_type == "Square Wave":
            center = st.slider("Pulse Center", 0.0, 1.0, 0.5, step=0.05)
            width = st.slider("Pulse Width", 0.1, 0.5, 0.2, step=0.05)
            u0_func = lambda x: np.where(np.abs((x - center + 0.5) % 1.0 - 0.5) < width/2, 1.0, 0.1)
        
        elif ic_type == "Triangle Wave":
            center = st.slider("Peak Position", 0.0, 1.0, 0.5, step=0.05)
            width = st.slider("Base Width", 0.1, 0.5, 0.3, step=0.05)
            def triangle(x):
                dist = np.abs((x - center + 0.5) % 1.0 - 0.5)
                return np.maximum(0, 1 - dist / (width/2))
            u0_func = lambda x: triangle(x) + 0.1
        
        else:  # Custom Function
            st.markdown("**Enter Python expression** (use `x` as variable)")
            custom_expr = st.text_input("Function u0(x)", "np.exp(-100*(x-0.3)**2)")
            try:
                test_x = np.linspace(0, 1, 10)
                test_result = eval(custom_expr, {"x": test_x, "np": np})
                u0_func = lambda x: eval(custom_expr, {"x": x, "np": np})
                st.success("✓ Valid function")
            except Exception as e:
                st.error(f"Invalid function: {e}")
                u0_func = lambda x: np.exp(-100 * (x - 0.3)**2)
    
    with col2:
        st.markdown("**Initial Condition Preview:**")
        preview_x = np.linspace(0, 1, 100, endpoint=False)
        preview_y = u0_func(preview_x)
        preview_y = preview_y / np.linalg.norm(preview_y)
        
        fig_preview, ax_preview = plt.subplots(figsize=(5, 3), facecolor='white')
        ax_preview.plot(preview_x, preview_y, 'b-', linewidth=2.5)
        ax_preview.fill_between(preview_x, preview_y, alpha=0.25, color='blue')
        ax_preview.set_xlabel("Position x", fontsize=10, fontweight='bold')
        ax_preview.set_ylabel("u₀(x)", fontsize=10, fontweight='bold')
        ax_preview.grid(True, alpha=0.2, linestyle='--')
        ax_preview.spines['top'].set_visible(False)
        ax_preview.spines['right'].set_visible(False)
        st.pyplot(fig_preview)
        plt.close(fig_preview)
    
    shots = st.number_input("Number of Shots", min_value=10000, max_value=500000, value=200000, step=50000)
    
    run_sim_btn = st.button("Run Quantum Simulation", type="primary")
    
    if run_sim_btn:
        st.subheader("Time Evolution: Wave Evolution")
        
        # Setup
        steps = sorted(time_steps_display)
        N = 2**n_qubits
        x_grid = np.linspace(0, 1, N, endpoint=False)
        u0_vals = u0_func(x_grid)
        A_matrix, dt = get_classical_matrix(N, nu, c)
        
        # Create initial plot
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
        
        # Plot initial condition (t=0)
        initial_y = u0_vals / np.linalg.norm(u0_vals)
        ax.plot(x_grid, initial_y, color=colors[0], linestyle='-', alpha=0.8, linewidth=2.5, label='t=0 (Initial)')
        ax.fill_between(x_grid, initial_y, alpha=0.2, color=colors[0])
        
        ax.set_xlabel("Position x", fontsize=12, fontweight='bold')
        ax.set_ylabel("Amplitude u(x,t)", fontsize=12, fontweight='bold')
        ax.set_title(f"Advection-Diffusion Evolution (n={n_qubits}, ν={nu}, c={c})", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, max(initial_y) * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create placeholder for dynamic updates
        plot_placeholder = st.empty()
        plot_placeholder.pyplot(fig)
        
        # Status container
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Track classical evolution incrementally
        v_classical = u0_vals.copy()
        current_step = 0
        
        # Simulate and update plot for each timestep
        for idx, t in enumerate(steps[1:], start=1):  # Skip t=0, already plotted
            status_text.markdown(f"**Computing timestep {t}...** ⏳")
            
            # Run quantum simulation for this timestep
            # [FIX] FORCE EVEN PARITY
            # 1. We always enforce an EVEN degree. This stabilizes the QSP solver.
            deg = int(t + 8)
            if deg % 2 != 0: 
                deg += 1
            
            try:
                # Get state vector
                state_vector = u0_vals / np.linalg.norm(u0_vals)
                
                # [FIX] SYMMETRIC TARGET
                # We target exp(t*(|x|-1)) instead of exp(t*(x-1)).
                # This creates a symmetric "U" shape (or "Gaussian-like" shape) on [-1, 1].
                # Since our operator eigenvalues are only in [0, 1], the negative side 
                # is irrelevant for physics but ESSENTIAL for the solver to find a solution.
                target_f = lambda x: np.exp(t * (np.abs(x) - 1))
                
                # Solve using the robust settings
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)
                
                # Build and run quantum circuit
                from qiskit import transpile
                from qiskit_aer import AerSimulator
                
                qc = QSVT_circuit_universal(phi_seq, n_qubits, nu, init_state=state_vector, measurement=False)
                physical_time = t * dt
                data_reg = qc.qregs[2]
                adv_gate = Advection_Gate(n_qubits, c, physical_time)
                qc.append(adv_gate, data_reg)
                
                qc.measure(qc.qregs[0], qc.cregs[0])
                qc.measure(qc.qregs[1], qc.cregs[1])
                qc.measure(qc.qregs[2], qc.cregs[2])
                
                backend = AerSimulator()
                tqc = transpile(qc, backend, optimization_level=0)
                counts = backend.run(tqc, shots=shots).result().get_counts()
                
                # Process results
                prob_dist = np.zeros(N)
                total_valid = 0
                for key, count in counts.items():
                    parts = key.split()
                    s, a, d = "", "", ""
                    for p in parts:
                        if len(p) == 1: s = p
                        elif len(p) == 2: a = p
                        elif len(p) == n_qubits: d = p
                    if s == '0' and a == '00':
                        prob_dist[int(d, 2)] += count
                        total_valid += count
                
                if total_valid > 0:
                    prob_dist = prob_dist / total_valid
                    yq = np.sqrt(prob_dist)
                    
                    # Update classical evolution
                    steps_to_take = t - current_step
                    if steps_to_take > 0:
                        A_gap = np.linalg.matrix_power(A_matrix, steps_to_take)
                        v_classical = A_gap @ v_classical
                        current_step = t
                    y_class = v_classical / np.linalg.norm(v_classical)
                    
                    # Exact solution
                    phys_time = t * dt
                    y_exact = exact_solution_fourier(u0_vals, phys_time, nu, c)
                    
                    # Add new curves to plot
                    col = colors[idx]
                    ax.plot(x_grid, y_exact, color=col, linestyle='-', alpha=0.4, linewidth=1.5)
                    ax.plot(x_grid, y_class, color=col, linestyle='--', alpha=0.6, linewidth=1.5)
                    ax.plot(x_grid, yq, 'o', color=col, markersize=5, label=f't={t}', markeredgecolor='white', markeredgewidth=0.5)
                    ax.fill_between(x_grid, yq, alpha=0.1, color=col)
                    
                    # Update legend
                    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
                    
                    # Update plot
                    plot_placeholder.pyplot(fig)
                    status_text.markdown(f"**Timestep {t} complete.** ({total_valid}/{shots} valid shots, {100*total_valid/shots:.1f}%)")
                    
                else:
                    status_text.markdown(f"**Timestep {t} failed.**")
            
            except Exception as e:
                status_text.markdown(f"**Error at timestep {t}:** {str(e)}")
            
            progress_bar.progress(idx / len(steps))
        
        # Final status
        status_text.markdown("**Simulation Complete!** 🎉")
        progress_bar.progress(1.0)
        
        # Add final legend with line styles
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='gray', lw=2, alpha=0.4, label='Exact (Fourier)'),
            Line2D([0], [0], color='gray', lw=2, linestyle='--', alpha=0.6, label='Classical (Finite Diff)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, markeredgecolor='white', markeredgewidth=0.5, label='Quantum (QSVT)')
        ]
        ax.legend(handles=custom_lines, loc='upper left', fontsize=10, framealpha=0.9)
        plot_placeholder.pyplot(fig)
        plt.close()

else:
    st.info("👆 First, compute the QSVT angles using the button in Step 4 above.")
