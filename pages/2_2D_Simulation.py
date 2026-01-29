import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from simulation import run_split_step_sim_2d, exact_solution_fourier_2d, get_classical_matrix_2d
from quantum import QSVT_circuit_2d, Advection_Gate_2d
from solvers import cvx_poly_coef, Angles_Fixed

st.set_page_config(page_title="QSVT PDE Solver - 2D", layout="wide")
st.title("2D Diffusion-Advection: Quantum Singular Value Transformation")

st.markdown(r"""
### Theory Snapshot (2D)

$$\partial_t u = \nu (\partial_x^2 u + \partial_y^2 u) - c_x \partial_x u - c_y \partial_y u$$

**Operator flow:**
$$u_0 \;\xrightarrow{\text{Block-encode }A}\; \text{QSVT }P(A) \;\xrightarrow{\text{QFT advection}}\; u(t)$$

- Grid: $n_x \times n_y$ with $n_x+n_y$ qubits
- Diffusion: 5-point Laplacian via LCU block-encoding
- Advection: phase shifts on $x$ and $y$ registers
""")

if '2d_angles_computed' not in st.session_state:
    st.session_state['2d_angles_computed'] = False
if '2d_phi_sequences' not in st.session_state:
    st.session_state['2d_phi_sequences'] = {}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Quantum Hardware")
    nx = st.slider("Grid Width (nx)", min_value=2, max_value=200, value=3, step=1)
    ny = st.slider("Grid Height (ny)", min_value=2, max_value=200, value=3, step=1)
    n_total = nx + ny
    
    max_qubits = 24
    
    if n_total > max_qubits:
        st.error(f"Grid too large. Total qubits: {n_total} exceeds max {max_qubits}.")
        st.info(f"Reduce grid sizes so nx + ny ≤ {max_qubits}. Example: nx=12, ny=12 (144 grid points, 2^24 state)")
        st.stop()
    
    st.caption(f"Total qubits: {n_total} (x:{nx}, y:{ny}) → {nx*ny} grid points")
    if n_total > 18:
        st.warning(f"Large grid: {n_total} qubits = 2^{n_total} = {2**n_total:,} state elements. Computation may be slow.")
    st.header("Physics Parameters")
    nu = st.slider("Viscosity (ν)", 0.005, 0.05, 0.02, step=0.001)
    c_x = st.slider("X-Advection Velocity (cₓ)", -1.0, 1.0, 0.3, step=0.05)
    c_y = st.slider("Y-Advection Velocity (cᵧ)", -1.0, 1.0, 0.3, step=0.05)
    
    st.header("Time Evolution")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t_max = st.slider("Max Time Steps", 10, 60, 30, step=10)
    with col_t2:
        n_timesteps = st.slider("Number of Time Steps", 3, 10, 5, step=1)
    
    time_steps_display = [int(i * t_max / (n_timesteps - 1)) for i in range(n_timesteps)]
    st.info(f"Visualizing {len(time_steps_display)} time steps: {time_steps_display}")

# --- MAIN CONTENT ---

st.header("Step 1: The 2D Advection-Diffusion Equation")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    We solve the 2D advection-diffusion PDE with periodic boundary conditions:
    """)
    st.latex(r"\frac{\partial u}{\partial t} = \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) - c_x \frac{\partial u}{\partial x} - c_y \frac{\partial u}{\partial y}")
    st.markdown("""
    Where:
    - $u(x,y,t)$ is the wave amplitude at position $(x,y)$ and time $t$
    - $\\nu$ is the **diffusion coefficient** (viscosity) - controls spreading
    - $c_x, c_y$ are the **advection velocities** - control transport
    - Domain: $(x,y) \\in [0, 1) \\times [0, 1)$ with periodic boundaries
    - Uses **2D Laplacian** operator: $\\nabla^2 u = \\partial_x^2 u + \\partial_y^2 u$
    """)

with col2:
    st.info(f"""
    **Current Parameters:**
    - ν = {nu}
    - cₓ = {c_x}
    - cᵧ = {c_y}
    - Grid: {nx}×{ny} = {nx*ny} points
    - dx = {1/nx:.4f}
    - dy = {1/ny:.4f}
    - dt ≈ {0.9*min(1/nx**2, 1/ny**2)/(4*nu):.5f}
    """)

st.header("Step 2: 2D Block-Encoded Laplacian")
st.markdown("""
The 2D diffusion operator uses a **5-point stencil** (center + 4 neighbors):
""")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
    ```
    Stencil:    [   0 ]
                [   1 ]
            [ 1 -4  1 ]
                [   1 ]
                [   0 ]
    ```
    
    - Center coefficient: **-4**
    - Each neighbor: **+1**
    - Encodes: $\\nabla^2 u = \\frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{(dx)^2}$
    """)

with col2:
    st.markdown(f"""
    **Block Encoding Strategy:**
    - Separate registers for x and y coordinates
    - Shift gates on both axes independently
    - Ancilla postselection projects onto Laplacian
    
    **Circuit Structure:**
    - Total qubits: {n_total + 3} (x: {nx}, y: {ny}, ancilla: 2, signal: 1)
    - Spatial resolution: {nx}×{ny}
    """)

st.header("Step 3: 2D QSVT Circuit")
st.markdown("""
**Quantum Singular Value Transformation** in 2D applies polynomial $P(\\nabla^2)$ to the Laplacian:
""")

col1, col2 = st.columns([1, 1])
with col1:
    st.latex(r"P(\nabla^2) = e^{(\nabla^2-I) \cdot t} \approx \sum_{k=0}^d c_k T_k(\nabla^2)")
    st.markdown("""
    - Chebyshev polynomial approximation
    - Degree $d$ increases with time step
    - Requires $d+1$ rotation angles
    """)

with col2:
    st.markdown(f"""
    **2D Circuit Complexity:**
    - Data qubits: {n_total}
    - Ancilla qubits: 2
    - Signal qubit: 1
    - **Total: {n_total + 3} qubits**
    
    **Scaling:**
    - 2×2 grid: 4 qubits ✓
    - 3×3 grid: 6 qubits ✓
    - 4×4 grid: 8 qubits ✓
    - 5×5 grid: 10 qubits (deep)
    """)

st.header("Step 4: Compute QSVT Angles")

calc_angles_btn = st.button("⚙️ Calculate Angles for Selected Time Steps", type="primary")

if calc_angles_btn:
    with st.spinner("Computing 2D QSVT angles..."):
        st.session_state['2d_phi_sequences'] = {}
        progress_bar = st.progress(0)
        
        for idx, t in enumerate(time_steps_display):
            if t == 0:
                continue
            
            deg = int(t + 8)
            if deg % 2 != 0:
                deg += 1
            
            try:
                target_f = lambda x: np.exp(t * (np.abs(x) - 1))
                coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
                phi_seq = Angles_Fixed(coef)
                st.session_state['2d_phi_sequences'][t] = phi_seq
            except Exception as e:
                st.error(f"Failed to compute angles for t={t}: {e}")
            
            progress_bar.progress((idx + 1) / len(time_steps_display))
        
        st.session_state['2d_angles_computed'] = True
        st.success(f"✓ Successfully computed angles for {len(st.session_state['2d_phi_sequences'])} time steps!")

if st.session_state['2d_angles_computed']:
    st.markdown("**Angle Sequence Information:**")
    angle_data = []
    for t, phi in st.session_state['2d_phi_sequences'].items():
        angle_data.append({
            "Time Step": t,
            "Polynomial Degree": len(phi) - 1,
            "Num Angles": len(phi),
            "First/Last Angle": f"[{phi[0]:.3f}, {phi[-1]:.3f}]"
        })
    st.dataframe(angle_data, use_container_width=True)

if st.session_state['2d_angles_computed']:
    st.header("Step 5: Choose Initial Condition and Simulate")
    
    # Display time-color legend
    st.markdown("### Time Step Color Legend")
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_steps_display)))
    legend_cols = st.columns(min(5, len(time_steps_display)))
    for idx, (t, col) in enumerate(zip(time_steps_display, colors)):
        with legend_cols[idx % len(legend_cols)]:
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(col[0]*255), int(col[1]*255), int(col[2]*255))
            st.markdown(f"<span style='color:{color_hex}'>■</span> t = {t}", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    u0_func = None
    
    with col1:
        ic_type = st.selectbox("2D Function Type", [
            "Gaussian Peak",
            "Double Gaussian",
            "Gaussian Ring",
            "Sine Pattern",
            "Custom Function"
        ])
        
        if ic_type == "Gaussian Peak":
            center_x = st.slider("Peak Center X", 0.0, 1.0, 0.5, step=0.05)
            center_y = st.slider("Peak Center Y", 0.0, 1.0, 0.5, step=0.05)
            width = st.slider("Peak Width", 1.0, 20.0, 10.0, step=0.5)
            
            st.session_state['ic_params'] = {'type': 'gaussian_peak', 'center_x': center_x, 'center_y': center_y, 'width': width}
            
            def u0_func():
                x = np.linspace(0, 1, nx, endpoint=False)
                y = np.linspace(0, 1, ny, endpoint=False)
                X, Y = np.meshgrid(x, y, indexing='ij')
                u0 = np.exp(-width * ((X - center_x)**2 + (Y - center_y)**2))
                norm = np.linalg.norm(u0.flatten())
                return u0 / norm if norm > 0 else u0
        
        elif ic_type == "Double Gaussian":
            x1, y1 = st.columns(2)
            with x1:
                c1x = st.slider("Peak 1 X", 0.0, 1.0, 0.3, step=0.05)
            with y1:
                c1y = st.slider("Peak 1 Y", 0.0, 1.0, 0.3, step=0.05)
            
            x2, y2 = st.columns(2)
            with x2:
                c2x = st.slider("Peak 2 X", 0.0, 1.0, 0.7, step=0.05)
            with y2:
                c2y = st.slider("Peak 2 Y", 0.0, 1.0, 0.7, step=0.05)
            
            width = st.slider("Peak Width", 1.0, 20.0, 8.0, step=0.5)
            
            st.session_state['ic_params'] = {'type': 'double_gaussian', 'c1x': c1x, 'c1y': c1y, 'c2x': c2x, 'c2y': c2y, 'width': width}
            
            def u0_func():
                x = np.linspace(0, 1, nx, endpoint=False)
                y = np.linspace(0, 1, ny, endpoint=False)
                X, Y = np.meshgrid(x, y, indexing='ij')
                u0 = (np.exp(-width * ((X - c1x)**2 + (Y - c1y)**2)) + 
                      np.exp(-width * ((X - c2x)**2 + (Y - c2y)**2)))
                norm = np.linalg.norm(u0.flatten())
                return u0 / norm if norm > 0 else u0
        
        elif ic_type == "Gaussian Ring":
            center_x = st.slider("Ring Center X", 0.0, 1.0, 0.5, step=0.05)
            center_y = st.slider("Ring Center Y", 0.0, 1.0, 0.5, step=0.05)
            radius = st.slider("Ring Radius", 0.1, 0.4, 0.2, step=0.05)
            width = st.slider("Ring Width", 1.0, 20.0, 10.0, step=0.5)
            
            def u0_func():
                x = np.linspace(0, 1, nx, endpoint=False)
                y = np.linspace(0, 1, ny, endpoint=False)
                X, Y = np.meshgrid(x, y, indexing='ij')
                r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                u0 = np.exp(-width * (r - radius)**2)
                norm = np.linalg.norm(u0.flatten())
                return u0 / norm if norm > 0 else u0
        
        elif ic_type == "Sine Pattern":
            freq_x = st.slider("Frequency X", 1, 5, 2)
            freq_y = st.slider("Frequency Y", 1, 5, 2)
            
            def u0_func():
                x = np.linspace(0, 1, nx, endpoint=False)
                y = np.linspace(0, 1, ny, endpoint=False)
                X, Y = np.meshgrid(x, y, indexing='ij')
                u0 = np.abs(np.sin(2 * np.pi * freq_x * X) * np.sin(2 * np.pi * freq_y * Y))
                norm = np.linalg.norm(u0.flatten())
                return u0 / norm if norm > 0 else u0
        
        else:
            st.markdown("**Enter Python expression** (use `X, Y` as variables)")
            custom_expr = st.text_input("Function u0(X,Y)", "np.exp(-10*((X-0.5)**2 + (Y-0.5)**2))")
            
            try:
                x_test = np.linspace(0, 1, 3)
                y_test = np.linspace(0, 1, 3)
                X_test, Y_test = np.meshgrid(x_test, y_test)
                test_result = eval(custom_expr, {"X": X_test, "Y": Y_test, "np": np})
                
                def u0_func():
                    x = np.linspace(0, 1, nx, endpoint=False)
                    y = np.linspace(0, 1, ny, endpoint=False)
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    u0 = eval(custom_expr, {"X": X, "Y": Y, "np": np})
                    norm = np.linalg.norm(u0.flatten())
                    return u0 / norm if norm > 0 else u0
                
                st.success("✓ Valid function")
            except Exception as e:
                st.error(f"Invalid function: {e}")
                
                def u0_func():
                    x = np.linspace(0, 1, nx, endpoint=False)
                    y = np.linspace(0, 1, ny, endpoint=False)
                    X, Y = np.meshgrid(x, y, indexing='ij')  # ij indexing
                    u0 = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))
                    norm = np.linalg.norm(u0.flatten())
                    return u0 / norm if norm > 0 else u0
    
    # Display preview after all u0_func definitions
    if u0_func is not None:
        with col2:
            st.markdown("**Initial Condition Preview:**")
            u0_init = u0_func()
            
            # Ensure data is real and properly scaled
            u0_display = np.abs(u0_init)
            u0_display = u0_display / (np.max(u0_display) + 1e-10)  # Normalize to [0,1]
            
            # Create a finer grid for smooth visualization (not blocky)
            # Compute on high-res grid for preview while keeping quantum computation on coarse grid
            x_fine = np.linspace(0, 1, max(128, nx*32), endpoint=False)
            y_fine = np.linspace(0, 1, max(128, ny*32), endpoint=False)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')  # ij indexing
            
            # Evaluate the same function on fine grid for smooth preview
            # Get the definition from ic_type
            if ic_type == "Gaussian Peak":
                u0_fine = np.exp(-width * ((X_fine - center_x)**2 + (Y_fine - center_y)**2))
            elif ic_type == "Double Gaussian":
                u0_fine = (np.exp(-width * ((X_fine - c1x)**2 + (Y_fine - c1y)**2)) + 
                           np.exp(-width * ((X_fine - c2x)**2 + (Y_fine - c2y)**2)))
            elif ic_type == "Gaussian Ring":
                r_fine = np.sqrt((X_fine - center_x)**2 + (Y_fine - center_y)**2)
                u0_fine = np.exp(-width * (r_fine - radius)**2)
            elif ic_type == "Sine Pattern":
                u0_fine = np.abs(np.sin(2 * np.pi * freq_x * X_fine) * np.sin(2 * np.pi * freq_y * Y_fine))
            else:  # Custom
                try:
                    u0_fine = eval(custom_expr, {"X": X_fine, "Y": Y_fine, "np": np})
                    u0_fine = np.abs(u0_fine)
                except:
                    u0_fine = np.exp(-10 * ((X_fine - 0.5)**2 + (Y_fine - 0.5)**2))
            
            # Normalize fine grid display
            u0_fine_display = np.abs(u0_fine)
            u0_fine_display = u0_fine_display / (np.max(u0_fine_display) + 1e-10)
            
            fig_preview, ax_preview = plt.subplots(figsize=(5, 4), facecolor='white')
            # Use interpolation to smooth the display
            im = ax_preview.imshow(u0_fine_display, cmap='viridis', origin='lower', 
                                   extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
            ax_preview.set_xlabel("X", fontsize=10, fontweight='bold')
            ax_preview.set_ylabel("Y", fontsize=10, fontweight='bold')
            ax_preview.set_title(f"u₀({nx}×{ny}) - Smooth Preview", fontsize=11, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax_preview, label='Normalized Amplitude')
            
            # Add value info from coarse grid
            st.caption(f"Min: {np.min(u0_display):.4f}, Max: {np.max(u0_display):.4f}, Mean: {np.mean(u0_display):.4f}")
            
            st.pyplot(fig_preview)
            plt.close(fig_preview)
    
    shots = st.number_input("Number of Shots", min_value=10000, max_value=200000, value=100000, step=10000)
    
    run_sim_btn = st.button("Run 2D Quantum Simulation", type="primary")
    
    if run_sim_btn:
        st.subheader("2D Time Evolution: Heatmap Visualization")
        
        # Setup
        steps = sorted(time_steps_display)
        
        # Validate grid size before allocation
        n_qubits = nx + ny
        max_qubits = 24
        if n_qubits > max_qubits:
            st.error(f"Grid too large. Total qubits: {n_qubits} exceeds max {max_qubits}.")
            st.info(f"Reduce grid sizes so nx + ny ≤ {max_qubits}. Example: nx=12, ny=12 (144 grid points, 2^24 state)")
            st.stop()
        
        dx = 1 / nx
        dy = 1 / ny
        dt = 0.9 * min(dx**2, dy**2) / (4 * nu)
        
        # Create figure with subplots for each timestep
        n_cols = min(3, len(steps))
        n_rows = (len(steps) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), facecolor='white')
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape((n_rows, n_cols))
        
        colors_map = plt.cm.viridis(np.linspace(0, 1, len(steps)))
        
        # Initialize classical evolution
        u0_2d = u0_func()
        u0_flat = u0_2d.flatten()
        
        # Pad to full 2^(nx+ny) state vector for quantum circuit
        full_state_dim = 2 ** n_qubits
        u0_full = np.zeros(full_state_dim)
        u0_full[:len(u0_flat)] = u0_flat
        u0_full = u0_full / np.linalg.norm(u0_full)
        
        A_matrix, dt = get_classical_matrix_2d(nx, ny, nu, c_x, c_y)
        
        v_classical = u0_flat.copy()
        current_step = 0
        
        # Status container
        status_text = st.empty()
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        
        # Simulate each timestep
        for step_idx, t in enumerate(steps):
            status_text.markdown(f"**Computing timestep {t}...** ⏳")
            
            row_idx = step_idx // n_cols
            col_idx = step_idx % n_cols
            ax = axes[row_idx, col_idx]
            
            if t == 0:
                # Initial condition - evaluate directly on fine grid for accurate visualization
                # Create fine grid
                n_fine = max(256, max(nx, ny)*32)
                x_fine = np.linspace(0, 1, n_fine, endpoint=False)
                y_fine = np.linspace(0, 1, n_fine, endpoint=False)
                X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
                
                # Evaluate initial condition directly on fine grid using stored parameters
                ic_params = st.session_state.get('ic_params', {})
                if ic_params.get('type') == 'gaussian_peak':
                    u0_fine = np.exp(-ic_params['width'] * ((X_fine - ic_params['center_x'])**2 + (Y_fine - ic_params['center_y'])**2))
                else:
                    # Fallback: use zoom interpolation
                    from scipy.ndimage import zoom
                    u0_fine = zoom(u0_2d, (n_fine/nx, n_fine/ny), order=1)
                
                u0_fine_norm = u0_fine / (np.max(u0_fine) + 1e-10)
                
                im = ax.imshow(u0_fine_norm, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], 
                              aspect='auto', interpolation='bilinear')
                ax.set_title(f"t = 0 (Initial)", fontsize=12, fontweight='bold')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                plt.colorbar(im, ax=ax)
                progress_bar.progress((step_idx + 1) / len(steps))
                continue
            
            try:
                # Compute angles (use stored angles if available)
                if t not in st.session_state['2d_phi_sequences']:
                    # Compute if not already cached
                    deg_diff = int(t + 8)
                    if deg_diff % 2 != 0:
                        deg_diff += 1
                    
                    target_f = lambda x: np.exp(t * (np.abs(x) - 1))
                    coef_diff = cvx_poly_coef(target_f, [0, 1], deg_diff, epsil=1e-5)
                    phi_seq = Angles_Fixed(coef_diff)
                    st.session_state['2d_phi_sequences'][t] = phi_seq
                else:
                    phi_seq = st.session_state['2d_phi_sequences'][t]
                
                # Build 2D quantum circuit
                qc = QSVT_circuit_2d(phi_seq, nx, ny, nu, init_state=u0_full, measurement=True)
                
                # Append advection before measurement
                physical_time = t * dt
                adv_gate = Advection_Gate_2d(nx, ny, c_x, c_y, physical_time)
                # Get x and y registers by name
                x_reg = None
                y_reg = None
                for qreg in qc.qregs:
                    if qreg.name == 'x':
                        x_reg = qreg
                    elif qreg.name == 'y':
                        y_reg = qreg
                if x_reg and y_reg:
                    qc.append(adv_gate, list(x_reg) + list(y_reg))
                
                # Run circuit
                from qiskit import transpile
                from qiskit_aer import AerSimulator
                
                backend = AerSimulator()
                tqc = transpile(qc, backend, optimization_level=0)
                job = backend.run(tqc, shots=shots)
                counts = job.result().get_counts()
                
                # Process measurement results
                # Parse measurement keys - they are space-separated registers
                prob_dist = np.zeros(nx * ny)
                total_valid = 0
                
                for bitstring, count in counts.items():
                    try:
                        # Measurement format from Qiskit: "m_dat m_anc m_sig"
                        # where m_dat is nx+ny bits, m_anc is 2 bits, m_sig is 1 bit
                        parts = bitstring.split()
                        
                        if len(parts) != 3:
                            continue
                        
                        m_dat = parts[0]
                        m_anc = parts[1]
                        m_sig = parts[2]
                        
                        # Validate lengths
                        if len(m_dat) != nx + ny or len(m_anc) != 2 or len(m_sig) != 1:
                            continue
                        
                        # Postselection: sig=0 and anc=00
                        if m_sig == '0' and m_anc == '00':
                            # Parse x and y from m_dat (first nx bits are x, next ny bits are y)
                            x_bits = m_dat[:nx]
                            y_bits = m_dat[nx:]
                            x_idx = int(x_bits, 2)
                            y_idx = int(y_bits, 2)
                            
                            # Convert to flat index: idx = x*ny + y for (nx, ny) shaped array
                            flat_idx = x_idx * ny + y_idx
                            
                            if 0 <= flat_idx < nx * ny:
                                prob_dist[flat_idx] += count
                                total_valid += count
                    except (ValueError, IndexError, AttributeError):
                        # Skip malformed strings
                        continue
                
                if total_valid > 0:
                    prob_dist = prob_dist / total_valid
                    # Reshape with proper 2D indexing: flat index = x + y*nx
                    # So reshape should give us (nx, ny) array first, not (ny, nx)
                    yq_2d = np.sqrt(prob_dist).reshape((nx, ny), order='C')
                    
                    # Update classical
                    steps_to_take = t - current_step
                    if steps_to_take > 0:
                        A_gap = np.linalg.matrix_power(A_matrix, steps_to_take)
                        v_classical = A_gap @ v_classical
                        current_step = t
                    
                    y_class = v_classical / np.linalg.norm(v_classical)
                    y_class_2d = y_class.reshape((nx, ny), order='C')  # Match quantum indexing
                    
                    # Exact
                    y_exact_2d = exact_solution_fourier_2d(u0_2d, physical_time, nu, c_x, c_y)
                    
                    # Plot quantum result - use smooth interpolation
                    from scipy.ndimage import zoom
                    yq_fine = zoom(yq_2d, (max(256, nx*32)/nx, max(256, ny*32)/ny), order=1)
                    
                    im = ax.imshow(yq_fine, cmap='viridis', origin='lower', extent=[0, 1, 0, 1],
                                  aspect='auto', interpolation='bilinear')
                    ax.set_title(f"t = {t} (Quantum, {100*total_valid/shots:.1f}%)", fontsize=12, fontweight='bold')
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    plt.colorbar(im, ax=ax)
                    
                    status_text.markdown(f"**Timestep {t} complete.**")
                else:
                    ax.text(0.5, 0.5, "Failed", ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
                    status_text.markdown(f"**Timestep {t} failed.**")
            
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
                status_text.markdown(f"**Error at timestep {t}:** {str(e)[:50]}")
            
            progress_bar.progress((step_idx + 1) / len(steps))
        
        # Hide unused subplots
        for idx in range(step_idx + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        status_text.markdown("**2D Simulation Complete!** 🎉")
        progress_bar.progress(1.0)

else:
    st.info("👆 First, compute the QSVT angles using the button in Step 4 above.")
