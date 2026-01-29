import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from solvers import cvx_poly_coef, Angles_Fixed
from quantum import QSVT_circuit_universal, Advection_Gate

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from solvers import cvx_poly_coef, Angles_Fixed
from quantum import QSVT_circuit_universal, Advection_Gate

def exact_solution_fourier(u0_vals, t_phys, nu, c):
    """Spectral method exact solution using FFT."""
    N = len(u0_vals)
    ak = np.fft.fft(u0_vals)
    k_vals = np.fft.fftfreq(N, d=1/N)
    decay = np.exp(-4 * (np.pi**2) * nu * (k_vals**2) * t_phys)
    phase = np.exp(-2j * np.pi * c * k_vals * t_phys)
    ak_new = ak * decay * phase
    u_exact = np.fft.ifft(ak_new).real
    return u_exact / np.linalg.norm(u_exact)

def get_classical_matrix(N, nu, c):
    """Central difference matrix for time stepping."""
    dx = 1/N
    dt = 0.9 * dx**2 / (2 * nu)
    alpha = dt * nu / (dx**2)
    gamma = dt * c / (2 * dx)
    val_from_left  = alpha + gamma
    val_from_right = alpha - gamma
    val_stay       = 1.0 - (val_from_left + val_from_right)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = val_stay
        A[i, (i-1)%N] = val_from_left
        A[i, (i+1)%N] = val_from_right
    return A, dt

def plot_combined_comparison(results, n, nu, c, u0_func):
    plt.figure(figsize=(12, 7))
    times = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    N = 2**n
    x_grid = np.linspace(0, 1, N, endpoint=False)
    u0_vals = u0_func(x_grid)
    A, dt = get_classical_matrix(N, nu, c)
    v_classical = u0_vals.copy()
    print("--- Plotting Comparison ---")
    print(f"dt (step) = {dt:.6f}")
    current_step = 0
    for i, t_target in enumerate(times):
        col = colors[i]
        x_q, p_q = results[t_target]
        y_quant = np.sqrt(p_q)
        steps_to_take = t_target - current_step
        if steps_to_take > 0:
            A_gap = np.linalg.matrix_power(A, steps_to_take)
            v_classical = A_gap @ v_classical
            current_step = t_target

        # Normalize
        y_class = v_classical / np.linalg.norm(v_classical)

        # --- C. EXACT ANALYTICAL (Solid Line) ---
        physical_time = t_target * dt
        y_exact = exact_solution_fourier(u0_vals, physical_time, nu, c)

        # --- PLOT COMMANDS ---
        # 1. Exact (Ground Truth)
        plt.plot(x_grid, y_exact, color=col, linestyle='-', linewidth=1.5, alpha=0.4,
                 label=f'Exact (t={t_target})' if i==0 else "")

        # 2. Classical (Approximation Target)
        plt.plot(x_grid, y_class, color=col, linestyle='--', linewidth=1.5, alpha=0.7,
                 label=f'Classical Matrix' if i==0 else "")

        # 3. Quantum (Simulation Result)
        plt.plot(x_q, y_quant, 'o', color=col, markersize=5, label=f'Quantum t={t_target}')

    plt.title(f"3-Way Comparison: Exact vs Classical vs Quantum\n(n={n}, nu={nu}, c={c})")
    plt.xlabel("Position x")
    plt.ylabel("Amplitude u(x,t)")

    # Custom Legend to reduce clutter
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=2, alpha=0.4, label='Exact (Fourier)'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.7, label='Classical (Finite Diff)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='Quantum Data')]
    plt.legend(handles=custom_lines, loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.show()

def run_split_step_sim(n, nu, c, time_steps, u0_func, shots=100000):
    N = 2**n
    results = {}

    x_grid = np.linspace(0, 1, N, endpoint=False)
    u0_vals = u0_func(x_grid)
    state_vector = u0_vals / np.linalg.norm(u0_vals)

    # Physics for Advection
    dx = 1/N
    dt = 0.9 * dx**2 / (2 * nu)

    print(f"--- Running Split-Step (QSVT Diff + Unitary Adv) ---")
    print(f"    n={n}, nu={nu}, c={c}, dt={dt:.5f}")

    for t in time_steps:
        print(f"Simulating t_step = {t}...", end=" ")

        if t == 0:
            p0 = u0_vals**2
            results[0] = (x_grid, p0/np.sum(p0))
            print("Done.")
            continue

        # --- PART A: Diffusion (QSVT) ---
# [FIX] FORCE EVEN PARITY
        # 1. We always enforce an EVEN degree. This stabilizes the QSP solver.
        deg = int(t + 8)
        if deg % 2 != 0: 
            deg += 1
        
        try:
            # [FIX] SYMMETRIC TARGET
            # We target exp(t*(|x|-1)) instead of exp(t*(x-1)).
            # This creates a symmetric "U" shape (or "Gaussian-like" shape) on [-1, 1].
            # Since our operator eigenvalues are only in [0, 1], the negative side 
            # is irrelevant for physics but ESSENTIAL for the solver to find a solution.
            
            target_f = lambda x: np.exp(t * (np.abs(x) - 1))
            
            # Solve using the robust settings
            coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
            phi_seq = Angles_Fixed(coef)
            
        except Exception as e:
            print(f"Solver Error at t={t}: {e}")
            continue

        # [FIX] Create circuit WITHOUT measurements
        qc = QSVT_circuit_universal(phi_seq, n, nu, init_state=state_vector, measurement=False)

        # --- PART B: Advection (Unitary Append) ---
        physical_time = t * dt
        data_reg = qc.qregs[2] # [sig, anc, dat]

        # Append Advection Gate
        adv_gate = Advection_Gate(n, c, physical_time)
        qc.append(adv_gate, data_reg)

        # [FIX] Manually add measurements now
        # We know exactly where the registers are because we built them
        qc.measure(qc.qregs[0], qc.cregs[0]) # Signal -> c_sig
        qc.measure(qc.qregs[1], qc.cregs[1]) # Ancilla -> c_anc
        qc.measure(qc.qregs[2], qc.cregs[2]) # Data -> c_dat

        # --- Run ---
        backend = AerSimulator()
        tqc = transpile(qc, backend, optimization_level=0)
        counts = backend.run(tqc, shots=shots).result().get_counts()

        # Process Results
        prob_dist = np.zeros(N)
        total_valid = 0
        for key, count in counts.items():
            parts = key.split()
            s, a, d = "", "", ""
            for p in parts:
                if len(p) == 1: s = p
                elif len(p) == 2: a = p
                elif len(p) == n: d = p

            if s == '0' and a == '00':
                prob_dist[int(d, 2)] += count
                total_valid += count

        if total_valid > 0:
            results[t] = (x_grid, prob_dist / total_valid)
            print("Success.")
        else:
            results[t] = (x_grid, np.zeros(N))
            print("Failed.")

    return results

def plot_diff_adv_amplitude(results, n, nu, c, u0_func):
    """Plots Amplitude u(x,t) with Classical Advection benchmark."""
    plt.figure(figsize=(10, 6))
    times = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    # Classical Matrix (Central Diff)
    N = 2**n
    dx = 1/N
    dt = 0.9 * dx**2 / (2 * nu)

    alpha = dt * nu / (dx**2)
    gamma = dt * c / (2 * dx)

    # Coefficients:
    # To move RIGHT, node i needs contribution from LEFT (i-1).
    # Coeff of (i-1) is (alpha + gamma)
    val_from_left  = alpha + gamma
    val_from_right = alpha - gamma
    val_stay       = 1.0 - (val_from_left + val_from_right)

    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = val_stay
        A[i, (i-1)%N] = val_from_left
        A[i, (i+1)%N] = val_from_right

    v0 = u0_func(np.linspace(0, 1, N, endpoint=False))

    for i, t in enumerate(times):
        x_q, p_q = results[t]
        y_q_amp = np.sqrt(p_q) # Prob -> Amplitude

        # Classical
        v_final = np.linalg.matrix_power(A, t) @ v0
        # Normalize classical for comparison (Quantum is normalized)
        y_true_amp = v_final / np.linalg.norm(v_final)

        plt.plot(x_q, y_true_amp, linestyle='--', color=colors[i], alpha=0.5)
        plt.plot(x_q, y_q_amp, 'o', color=colors[i], label=f't={t}', markersize=4)

    plt.title(f"Advection-Diffusion (n={n}, c={c})\nAmplitude u(x,t) [Split-Step]")
    plt.xlabel("Position x")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
# ==============================================================================
# 2D SIMULATIONS
# ==============================================================================

def exact_solution_fourier_2d(u0_2d, t_phys, nu, c_x, c_y):
    """
    2D spectral solution using FFT2.
    u0_2d: 2D array (nx, ny) with ij indexing
    Returns: normalized 2D solution with same shape
    """
    # FFT to wavenumber space
    u0_hat = np.fft.fft2(u0_2d)
    
    # Wavenumbers
    nx, ny = u0_2d.shape
    kx = np.fft.fftfreq(nx, d=1/nx)
    ky = np.fft.fftfreq(ny, d=1/ny)
    
    # 2D mesh - use ij indexing to match input
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    k_sq = Kx**2 + Ky**2
    
    # Evolution operators
    decay = np.exp(-4 * (np.pi**2) * nu * k_sq * t_phys)
    phase = np.exp(-2j * np.pi * (c_x * Kx + c_y * Ky) * t_phys)
    
    # Evolve and inverse FFT
    u_hat_new = u0_hat * decay * phase
    u_exact_2d = np.fft.ifft2(u_hat_new).real
    
    # Normalize
    u_norm = np.linalg.norm(u_exact_2d)
    return u_exact_2d / u_norm if u_norm > 0 else u_exact_2d

def get_classical_matrix_2d(nx, ny, nu, c_x, c_y):
    """
    2D finite difference matrix for advection-diffusion.
    Flattened indexing: idx = x + y*nx
    Returns: matrix A and timestep dt
    """
    N = nx * ny
    dx = 1 / nx
    dy = 1 / ny
    dt = 0.9 * min(dx**2, dy**2) / (4 * nu)  # 4 neighbors in 2D
    
    alpha_x = dt * nu / (dx**2)
    alpha_y = dt * nu / (dy**2)
    gamma_x = dt * c_x / (2 * dx)
    gamma_y = dt * c_y / (2 * dy)
    
    A = np.zeros((N, N))
    
    for y in range(ny):
        for x in range(nx):
            idx = x + y * nx
            
            # Center coefficient
            center_coef = 1.0 - 2*(alpha_x + alpha_y)
            A[idx, idx] = center_coef
            
            # X-neighbors (with advection)
            x_left = (x - 1) % nx
            x_right = (x + 1) % nx
            left_idx = x_left + y * nx
            right_idx = x_right + y * nx
            
            A[idx, left_idx] = alpha_x + gamma_x
            A[idx, right_idx] = alpha_x - gamma_x
            
            # Y-neighbors (with advection)
            y_down = (y - 1) % ny
            y_up = (y + 1) % ny
            down_idx = x + y_down * nx
            up_idx = x + y_up * nx
            
            A[idx, down_idx] = alpha_y + gamma_y
            A[idx, up_idx] = alpha_y - gamma_y
    
    return A, dt

def run_split_step_sim_2d(nx, ny, nu, c_x, c_y, time_steps, u0_2d_func, shots=100000):
    """
    2D split-step simulation with QSVT for diffusion + unitary for advection.
    u0_2d_func: function that returns 2D initial condition array (ny, nx)
    """
    N = nx * ny
    results = {}
    
    # Initial condition
    u0_2d = u0_2d_func()  # Shape: (ny, nx)
    u0_flat = u0_2d.flatten()
    state_vector = u0_flat / np.linalg.norm(u0_flat)
    
    # Physics
    dx = 1 / nx
    dy = 1 / ny
    dt = 0.9 * min(dx**2, dy**2) / (4 * nu)
    
    print(f"--- Running 2D Split-Step (QSVT Laplacian + Unitary Advection) ---")
    print(f"    nx={nx}, ny={ny}, nu={nu}, c_x={c_x}, c_y={c_y}, dt={dt:.5f}")
    
    for t in time_steps:
        print(f"Simulating t_step = {t}...", end=" ")
        
        if t == 0:
            p0_flat = u0_flat**2
            p0_2d = p0_flat.reshape((ny, nx))
            results[0] = p0_2d / np.sum(p0_2d)
            print("Done.")
            continue
        
        # Compute angles
        deg = int(t + 8)
        if deg % 2 != 0:
            deg += 1
        
        try:
            # Symmetric target for 2D
            target_f = lambda x: np.exp(t * (np.abs(x) - 1))
            coef = cvx_poly_coef(target_f, [0, 1], deg, epsil=1e-5)
            phi_seq = Angles_Fixed(coef)
        except Exception as e:
            print(f"Solver Error at t={t}: {e}")
            results[t] = np.zeros((ny, nx))
            continue
        
        try:
            # Import quantum circuit functions
            from quantum import QSVT_circuit_2d, Advection_Gate_2d
            
            # Build circuit
            qc = QSVT_circuit_2d(phi_seq, nx, ny, nu, init_state=state_vector, measurement=False)
            
            # Append 2D advection
            physical_time = t * dt
            x_reg = qc.qregs[1]
            y_reg = qc.qregs[2]
            adv_gate = Advection_Gate_2d(nx, ny, c_x, c_y, physical_time)
            qc.append(adv_gate, x_reg[:] + y_reg[:])
            
            # Measurements
            qc.measure(qc.qregs[0], qc.cregs[0])  # sig
            qc.measure(qc.qregs[1], qc.cregs[1])  # anc
            qc.measure(x_reg, qc.cregs[2][:nx])
            qc.measure(y_reg, qc.cregs[2][nx:])
            
            # Run
            backend = AerSimulator()
            tqc = transpile(qc, backend, optimization_level=0)
            counts = backend.run(tqc, shots=shots).result().get_counts()
            
            # Process results (2D)
            prob_dist_flat = np.zeros(N)
            total_valid = 0
            
            for key, count in counts.items():
                parts = key.split()
                sig_bit = ""
                anc_bits = ""
                data_bits = ""
                
                for p in parts:
                    if len(p) == 1:
                        sig_bit = p
                    elif len(p) == 2:
                        anc_bits = p
                    else:
                        data_bits = p
                
                if sig_bit == '0' and anc_bits == '00':
                    # Reconstruct 2D index from flattened measurement
                    flat_idx = int(data_bits, 2) if data_bits else 0
                    if 0 <= flat_idx < N:
                        prob_dist_flat[flat_idx] += count
                    total_valid += count
            
            if total_valid > 0:
                prob_dist_flat = prob_dist_flat / total_valid
                prob_dist_2d = prob_dist_flat.reshape((ny, nx))
                results[t] = prob_dist_2d
                print("Success.")
            else:
                results[t] = np.zeros((ny, nx))
                print("Failed (no valid shots).")
        
        except Exception as e:
            print(f"Error at t={t}: {e}")
            results[t] = np.zeros((ny, nx))
    
    return results