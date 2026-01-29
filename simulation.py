import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from solvers import cvx_poly_coef, Angles_Fixed
from quantum import QSVT_circuit_universal, Advection_Gate

# ==========================================
# 1. EXACT SOLUTION (Fourier Analysis)
# ==========================================
def exact_solution_fourier(u0_vals, t_phys, nu, c):
    """
    Computes Eq (6): u(x,t) using FFT spectral method.
    Exact for periodic boundaries.
    """
    N = len(u0_vals)
    # 1. Spatial -> Frequency Domain
    ak = np.fft.fft(u0_vals)

    # 2. Wavenumbers (k integers)
    k_vals = np.fft.fftfreq(N, d=1/N)

    # 3. Time Evolution Operators
    # Diffusion (Decay): exp(-4 * pi^2 * nu * k^2 * t)
    decay = np.exp(-4 * (np.pi**2) * nu * (k_vals**2) * t_phys)
    # Advection (Phase): exp(-2 * pi * i * c * k * t)
    phase = np.exp(-2j * np.pi * c * k_vals * t_phys)

    # 4. Evolve and Inverse FFT
    ak_new = ak * decay * phase
    u_exact = np.fft.ifft(ak_new).real

    return u_exact / np.linalg.norm(u_exact)

# ==========================================
# 2. CLASSICAL SOLUTION (Finite Difference)
# ==========================================
def get_classical_matrix(N, nu, c):
    """
    Constructs the Central Difference Matrix A.
    This represents the "Target Operator" the quantum computer approximates.
    """
    dx = 1/N
    dt = 0.9 * dx**2 / (2 * nu)

    alpha = dt * nu / (dx**2)
    gamma = dt * c / (2 * dx)

    # Matrix Coefficients
    val_from_left  = alpha + gamma   # Coeff for u[i-1]
    val_from_right = alpha - gamma   # Coeff for u[i+1]
    val_stay       = 1.0 - (val_from_left + val_from_right)

    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = val_stay
        A[i, (i-1)%N] = val_from_left
        A[i, (i+1)%N] = val_from_right

    return A, dt

# ==========================================
# 3. COMBINED PLOTTER
# ==========================================
def plot_combined_comparison(results, n, nu, c, u0_func):
    plt.figure(figsize=(12, 7))
    times = sorted(results.keys())

    # Use distinct colors for time steps
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    # Setup Classical & Exact Baselines
    N = 2**n
    x_grid = np.linspace(0, 1, N, endpoint=False)
    u0_vals = u0_func(x_grid)

    # Get Matrix A for Classical Step
    A, dt = get_classical_matrix(N, nu, c)
    v_classical = u0_vals.copy() # Will iterate this vector

    print("--- Plotting Comparison ---")
    print(f"dt (step) = {dt:.6f}")

    # We need to track the current step to evolve classical matrix correctly
    current_step = 0

    for i, t_target in enumerate(times):
        col = colors[i]

        # --- A. QUANTUM (Dots) ---
        x_q, p_q = results[t_target]
        y_quant = np.sqrt(p_q)

        # --- B. CLASSICAL MATRIX (Dashed Line) ---
        # Evolve matrix from current_step to t_target
        steps_to_take = t_target - current_step
        if steps_to_take > 0:
            # Efficiently apply matrix power for the gap
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
