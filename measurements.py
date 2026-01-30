import numpy as np
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Initialize

# Import from your codebase
from solvers import cvx_poly_coef, robust_poly_coef, Angles_Fixed
from quantum import Block_encoding_cosine, QSVT_circuit_1_ancilla

# ==============================================================================
# 1. DATA LOADING & HELPERS
# ==============================================================================

def get_function_data(func_name, n):
    """Generates normalized state vector for f(x)."""
    N = 2**n
    x = np.linspace(0, 1, N, endpoint=False)
    
    if func_name == "sin": 
        y = np.sin(2*np.pi*x) + 2.0
    elif func_name == "gaussian": 
        y = np.exp(-20*(x-0.5)**2)
    elif func_name == "linear": 
        y = 2*x + 0.5
    else: 
        y = np.ones(N)
        
    norm = np.linalg.norm(y)
    return y/norm, norm, x

# ==============================================================================
# 2. METHOD 1: SPECIAL INTERVALS (Compute-Uncompute)
# ==============================================================================

def get_window_state_circuit(n, interval_type):
    """Efficient window states using O(n) gates."""
    qc = QuantumCircuit(n, name=f"Window_{interval_type}")
    msb = n - 1
    
    if interval_type == "left_half": 
        for i in range(n):
            if i != msb: qc.h(i)
            
    elif interval_type == "middle_half": 
        msb_minus_1 = n - 2
        qc.h(msb)
        qc.cx(msb, msb_minus_1)
        qc.x(msb_minus_1)
        for i in range(msb_minus_1): 
            qc.h(i)
            
    elif interval_type == "full":
         for i in range(n): qc.h(i)
         
    return qc

def run_overlap_integral(n, func_name, interval_type, shots=10000):
    f_vec, f_norm, x_grid = get_function_data(func_name, n)
    
    qc_f = QuantumCircuit(n, name="U_f")
    qc_f.initialize(f_vec, range(n))
    qc_win = get_window_state_circuit(n, interval_type)
    
    full_qc = QuantumCircuit(n)
    full_qc.compose(qc_f, inplace=True)
    full_qc.compose(qc_win.inverse(), inplace=True)
    full_qc.measure_all()
    
    sim = AerSimulator()
    counts = sim.run(transpile(full_qc, sim), shots=shots).result().get_counts()
    
    p_zero = counts.get('0'*n, 0) / shots
    overlap = np.sqrt(p_zero)
    
    N = 2**n
    num_points = N if interval_type == "full" else N / 2
    d_norm = np.sqrt(num_points)
    integral_est = overlap * f_norm * d_norm * (1/N)
    
    dx = 1/N
    if interval_type == "left_half": mask = (x_grid < 0.5)
    elif interval_type == "middle_half": mask = (x_grid >= 0.25) & (x_grid < 0.75)
    else: mask = np.ones_like(x_grid, dtype=bool)
    exact_integral = np.sum((f_vec * f_norm)[mask]) * dx
    
    decomposed = qc_win.decompose()
    return {
        "integral_est": integral_est, 
        "integral_exact": exact_integral,
        "error": abs(integral_est - exact_integral), 
        "gate_count_window": decomposed.count_ops(), 
        "depth_window": decomposed.depth()
    }

# ==============================================================================
# 3. METHOD 2: ARBITRARY INTERVALS (QSVT Parity Decomp)
# ==============================================================================
# 
# IMPORTANT BUG NOTE (see QSVT_EVEN_POLYNOMIAL_BUG.md for details):
# The QSVT_circuit_1_ancilla function has a fundamental issue with even polynomials.
# - Odd polynomials: Work correctly, amplitude = P(λ) × input
# - Even polynomials: Give sqrt((1+P)/2) instead of P directly
# 
# For symmetric intervals (like [0.25, 0.75]), the boxcar is symmetric around λ=0,
# causing P_odd ≈ 0 and P_even to carry all information. Since even polynomial
# doesn't work correctly, results for symmetric intervals will have ~30% error.
#
# Workarounds:
# 1. Use Method 1 (compute-uncompute) for special intervals
# 2. Use asymmetric intervals where P_odd contributes significantly
# ==============================================================================

def get_boxcar_targets(a, b):
    """Defines Even/Odd parts of the boxcar filter on eigenvalues."""
    l1, l2 = np.cos(np.pi*b), np.cos(np.pi*a)
    l_min, l_max = min(l1, l2), max(l1, l2)
    
    # TUNED PARAMETERS:
    # k=20: Optimal sharpness for stable approximation
    # scale=0.97: High enough for accuracy, low enough for stability
    k = 20
    scale = 0.97
    
    # Smooth Boxcar
    boxcar = lambda x: scale * 0.25 * (1 + np.tanh(k*(x - l_min))) * (1 - np.tanh(k*(x - l_max)))
    
    # Parity decomposition with 0.5 factor (REQUIRED to keep |Poly| <= 1)
    t_even = lambda x: 0.5 * (boxcar(x) + boxcar(-x))
    t_odd = lambda x: 0.5 * (boxcar(x) - boxcar(-x))
    
    return t_even, t_odd, scale

def run_qsvt_component(n, f_vec, phi_seq, shots, is_odd=False):
    """Runs a single parity component with proper post-selection.
    
    Post-selects on the appropriate ancilla state based on polynomial parity:
    - Even polynomials: sig=0, anc=0
    - Odd polynomials: sig=1, anc=1
    
    Using the 1-ancilla circuit optimized for Cosine Block Encoding.
    """
    if phi_seq is None or len(phi_seq) == 0:
        return 0.0

    be_gate = Block_encoding_cosine(n).to_gate()
    
    # Build Circuit WITHOUT measurements yet
    qc = QSVT_circuit_1_ancilla(phi_seq, n, be_gate, init_state=f_vec, measurement=False)
    
    # Apply Hadamard to data to measure overlap with uniform superposition
    dat_reg = qc.qregs[2] 
    qc.h(dat_reg)
    
    # Use explicit measurements into pre-defined classical registers
    # This ensures predictable bitstring format: "m_dat m_anc m_sig"
    sig_reg = qc.qregs[0]
    anc_reg = qc.qregs[1]
    c_sig = qc.cregs[0]
    c_anc = qc.cregs[1]
    c_dat = qc.cregs[2]
    
    qc.measure(sig_reg, c_sig)
    qc.measure(anc_reg, c_anc)
    qc.measure(dat_reg, c_dat)
    
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    
    # Post-selection based on parity:
    # For this QSVT circuit (H-Rz-U-CRz-...-H structure):
    # - Even polynomials: sig=0, anc=0 (even number of angles)
    # - Odd polynomials: sig=1, anc=0 (odd number of angles)
    # The ancilla is always post-selected on 0 (block encoding success)
    target_sig = '1' if is_odd else '0'
    target_anc = '0'  # Always post-select on anc=0
    
    # Bitstring format: "m_dat m_anc m_sig" with spaces
    prob_valid = 0.0
    for bitstring, count in counts.items():
        parts = bitstring.split()
        if len(parts) == 3:
            m_dat, m_anc, m_sig = parts
            if m_sig == target_sig and m_anc == target_anc:
                if all(c == '0' for c in m_dat):
                    prob_valid += count
    
    prob = prob_valid / shots
    return np.sqrt(prob)

def solve_and_get_angles(target_func, deg):
    """Robust wrapper for solver + angle finding with optimized tolerances."""
    try:
        # epsil=2e-3 provides better numerical behavior than 1e-3
        # Use robust_poly_coef (NumPy-based) for Problem 2 stability
        coef = robust_poly_coef(target_func, [-1, 1], deg, epsil=2e-3)
        
        if coef is None or np.any(np.isnan(coef)):
            return None, False
        
        # FIX: Check for effectively zero polynomial to avoid solver crash
        # This happens (e.g.) for the Odd component when the interval is symmetric
        if np.linalg.norm(coef) < 1e-6:
             return [], False
            
        # Safety scaling 0.995 improves numerical stability/convergence
        coef_safe = coef * 0.995 
        phi = Angles_Fixed(coef_safe)
        return phi, True
        
    except Exception:
        return None, False

def run_qsvt_integral_arbitrary(n, func_name, a, b, shots=10000):
    f_vec, f_norm, x_grid = get_function_data(func_name, n)
    t_even, t_odd, boxcar_scale = get_boxcar_targets(a, b)
    
    # Degree selection
    # Adaptive degree based on grid resolution to avoid overfitting/instability
    if n <= 3:
        deg_even, deg_odd = 22, 23
    elif n == 4:
        deg_even, deg_odd = 20, 21
    elif n == 5:
        deg_even, deg_odd = 20, 21
    else:
        deg_even, deg_odd = 18, 19
    
    # --- EVEN COMPONENT ---
    phi_even, success_even = solve_and_get_angles(t_even, deg_even)
    if success_even and len(phi_even) > 0:
        amp_even = run_qsvt_component(n, f_vec, phi_even, shots, is_odd=False)
        val_even = (amp_even / boxcar_scale) * f_norm / np.sqrt(2**n)
    else:
        val_even = 0.0

    # --- ODD COMPONENT ---
    phi_odd, success_odd = solve_and_get_angles(t_odd, deg_odd)
    if success_odd and len(phi_odd) > 0:
        amp_odd = run_qsvt_component(n, f_vec, phi_odd, shots, is_odd=True)
        val_odd = (amp_odd / boxcar_scale) * f_norm / np.sqrt(2**n)
    else:
        val_odd = 0.0
    
    est = val_even + val_odd
    
    dx = 1/2**n
    mask = (x_grid >= a) & (x_grid <= b)
    exact = np.sum(f_vec * f_norm * mask) * dx
    
    return {
        "integral_est": est, 
        "integral_exact": exact, 
        "error": abs(est-exact),
        "val_even": val_even, 
        "val_odd": val_odd, 
        "deg_even": deg_even, 
        "deg_odd": deg_odd
    }


# ==============================================================================
# 4. METHOD 3: ARITHMETIC/COMPARISON APPROACH (Arbitrary Intervals)
# ==============================================================================

def _build_state_prep_unitary(target_state):
    """
    Build a unitary U such that U|0⟩ = |target_state⟩.
    Uses Gram-Schmidt orthogonalization.
    """
    N = len(target_state)
    
    # Start building orthonormal basis with target_state as first column
    U = np.zeros((N, N), dtype=complex)
    U[:, 0] = target_state
    
    # Gram-Schmidt for remaining columns
    for i in range(1, N):
        # Start with standard basis vector e_i
        v = np.zeros(N, dtype=complex)
        v[i] = 1.0
        
        # Orthogonalize against all previous columns
        for j in range(i):
            v = v - np.dot(np.conj(U[:, j]), v) * U[:, j]
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            U[:, i] = v / norm
        else:
            # If zero (linearly dependent), try other basis vectors
            for k in range(N):
                if k != i:
                    v = np.zeros(N, dtype=complex)
                    v[k] = 1.0
                    for j in range(i):
                        v = v - np.dot(np.conj(U[:, j]), v) * U[:, j]
                    norm = np.linalg.norm(v)
                    if norm > 1e-10:
                        U[:, i] = v / norm
                        break
    
    return U


def run_arithmetic_integral(n, func_name, a, b, shots=10000):
    """
    Computes the integral over [a, b] using the arithmetic/comparison approach.
    
    Algorithm (Compute-Uncompute style):
    1. Prepare uniform superposition |+⟩ = (1/√N) Σ_j |j⟩
    2. Mark states j in [a_int, b_int] (flip ancilla to |1⟩)  
    3. Apply U_f† (inverse of state preparation)
    4. Measure: P(main=0, mark=1) = (M/N) × |⟨χ_D|f⟩|²
    
    The integral is then computed from the overlap.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import UnitaryGate
    
    f_vec, f_norm, x_grid = get_function_data(func_name, n)
    N = 2**n
    
    # Convert bounds to integer indices
    a_int = max(0, min(int(np.floor(a * N)), N - 1))
    b_int = max(0, min(int(np.floor(b * N)), N - 1))
    if b_int < a_int:
        a_int, b_int = b_int, a_int
    
    num_points = b_int - a_int + 1
    
    main = QuantumRegister(n, 'main')
    mark = QuantumRegister(1, 'mark')
    c_main = ClassicalRegister(n, 'c_main')
    c_mark = ClassicalRegister(1, 'c_mark')
    
    qc = QuantumCircuit(main, mark, c_main, c_mark)
    
    # Step 1: Prepare uniform superposition
    qc.h(main)
    qc.barrier()
    
    # Step 2: Mark states in interval [a_int, b_int]
    # IMPORTANT: Qiskit's mcx ctrl_state convention:
    # ctrl_state[i] controls main[i], string is NOT reversed
    for j in range(a_int, b_int + 1):
        # Binary representation: LSB is main[0]
        ctrl_state = format(j, f'0{n}b')  # Big-endian binary string
        qc.mcx(main[:], mark[0], ctrl_state=ctrl_state)
    
    qc.barrier()
    
    # Step 3: Apply U_f† using UnitaryGate (workaround for StatePreparation.inverse() segfault)
    U_f = _build_state_prep_unitary(f_vec)
    U_f_dag = np.conj(U_f.T)
    qc.append(UnitaryGate(U_f_dag, label='U_f†'), main[:])
    
    qc.barrier()
    
    # Step 4: Measure
    qc.measure(main, c_main)
    qc.measure(mark, c_mark)
    
    # Run simulation
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim, optimization_level=3), shots=shots).result().get_counts()
    
    # Process results
    prob_zero_and_marked = 0
    prob_marked = 0
    
    for bitstring, count in counts.items():
        parts = bitstring.split()
        if len(parts) == 2:
            c_mark_str = parts[0]
            c_main_str = parts[1]
        else:
            c_mark_str = bitstring[0]
            c_main_str = bitstring[1:]
        
        if c_mark_str == '1':
            prob_marked += count
            if all(c == '0' for c in c_main_str):
                prob_zero_and_marked += count
    
    prob_marked /= shots
    prob_zero_and_marked /= shots
    
    # Derive overlap: P(main=0, mark=1) = (M/N) × |⟨χ_D|f⟩|²
    if num_points > 0:
        overlap_sq = prob_zero_and_marked * N / num_points
        overlap = np.sqrt(max(0, overlap_sq))
    else:
        overlap = 0.0
    
    # Integral estimation: I = ⟨χ_D|f⟩ × ||f|| × ||χ_D|| × dx
    dx = 1.0 / N
    d_norm = np.sqrt(num_points)
    integral_est = overlap * f_norm * d_norm * dx
    
    # Exact integral
    mask = (x_grid >= a) & (x_grid < b)
    exact_integral = np.sum((f_vec * f_norm)[mask]) * dx
    
    decomposed = transpile(qc, sim, optimization_level=0)
    
    return {
        "integral_est": integral_est,
        "integral_exact": exact_integral,
        "error": abs(integral_est - exact_integral),
        "interval_int": f"[{a_int}, {b_int}]",
        "num_points": num_points,
        "post_select_rate": prob_marked,
        "p_zero_and_marked": prob_zero_and_marked,
        "overlap": overlap,
        "depth": decomposed.depth(),
        "gate_count": sum(decomposed.count_ops().values())
    }


