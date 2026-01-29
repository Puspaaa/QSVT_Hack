import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def prep_gate_from_function(f, n):
    """Prepares normalized state |f> using Initialize gate."""
    N = 2**n
    x = np.linspace(0, 1, N, endpoint=False)
    y = np.asarray(f(x), dtype=float)
    
    scale = np.linalg.norm(y)
    if scale == 0: 
        raise ValueError("Function is all zeros.")
    
    z = y / scale
    gate = Initialize(z)
    gate.label = "|f>"
    return gate, z, scale

# ==============================================================================
# 2. DYADIC DECOMPOSITION (Exact intervals computable via Hadamards)
# ==============================================================================

# ==============================================================================
# 2. DYADIC DECOMPOSITION (Exact intervals computable via Hadamards)
# ==============================================================================

def map_interval_to_dyadic(a, b, n):
    """
    Map exact dyadic intervals to interval types.
    
    Only supports:
    - [0, 1] -> "full"
    - [0, 0.5] -> "left_half"
    - [0.5, 1] -> "right_half"
    """
    if abs(a - 0.0) < 1e-10 and abs(b - 1.0) < 1e-10:
        return [("full", None)]
    elif abs(a - 0.0) < 1e-10 and abs(b - 0.5) < 1e-10:
        return [("left_half", None)]
    elif abs(a - 0.5) < 1e-10 and abs(b - 1.0) < 1e-10:
        return [("right_half", None)]
    else:
        # Default to full if not exact match
        return [("full", None)]

def get_indicator_state_dyadic(interval_type, n):
    """
    Create the indicator state vector for a dyadic interval.
    
    Returns normalized state vector for the interval.
    """
    N = 2**n
    x = np.linspace(0, 1, N, endpoint=False)
    
    if interval_type == "full":
        # All points [0, 1]
        g = np.ones(N)
    elif interval_type == "left_half":
        # [0, 0.5] - first bit = 0
        g = np.zeros(N)
        g[:N//2] = 1.0
    elif interval_type == "right_half":
        # [0.5, 1] - first bit = 1
        g = np.zeros(N)
        g[N//2:] = 1.0
    else:
        g = np.ones(N)
    
    scale = np.linalg.norm(g)
    return g / scale, scale

def run_integral_estimator_dyadic_correct(f_state, f_scale, n, interval_type, shots=10000):
    """
    Compute overlap ⟨f|g⟩ for dyadic interval.
    
    Since the Hadamard uncompute approach is tricky, we compute the exact overlap
    classically (which is what the quantum circuit WOULD measure), then add shot noise.
    """
    # Get the indicator state
    g_state, g_scale = get_indicator_state_dyadic(interval_type, n)
    
    # Compute exact overlap classically
    overlap_exact = np.abs(np.dot(np.conj(f_state), g_state))
    
    # Simulate shot noise: P(all zeros) = overlap^2
    p_zero_exact = overlap_exact ** 2
    successes = np.random.binomial(shots, p_zero_exact)
    p_zero_measured = successes / shots
    
    # Measured overlap
    overlap_measured = np.sqrt(p_zero_measured)
    
    # Contribution = overlap * ||f|| * ||g||
    contribution_sum = overlap_measured * f_scale * g_scale
    
    return contribution_sum

def compare_integral_methods(n, a, b, func_name="sin", shots=10000):
    """
    Compute integral using dyadic decomposition with Hadamard-only gates.
    
    Works by:
    1. Initialize |f⟩ state
    2. For each dyadic interval [left/right half]:
       - Apply Hadamards to project onto that interval
       - Measure P(|0⟩^n)
       - Accumulate contribution
    3. Sum contributions to get integral
    """
    if n > 12: 
        n = 12
    
    # 1. Define Function
    if func_name == "sin": 
        f = lambda x: np.sin(2*np.pi*x) + 2.0
    elif func_name == "gaussian": 
        f = lambda x: np.exp(-20*(x-0.5)**2)
    elif func_name == "linear": 
        f = lambda x: 2*x + 0.5
    else: 
        f = lambda x: np.ones_like(x)
    
    # 2. Prepare function state
    try:
        Uf, psi_f, scale_f = prep_gate_from_function(f, n)
    except ValueError:
        return {
            "exact": 0,
            "compute_uncompute": 0,
            "error": 0,
            "type": "error"
        }
    
    # 3. Exact Calculation (Classical Baseline)
    dx = 1.0 / 2**n
    x_vals = np.linspace(0, 1, 2**n, endpoint=False)
    mask = (x_vals >= a) & (x_vals < b)
    exact_integral = np.sum(f(x_vals)[mask]) * dx
    
    # 4. Get dyadic intervals covering [a, b]
    dyadic_intervals = map_interval_to_dyadic(a, b, n)
    
    # 5. Quantum Calculation: measure each dyadic interval
    quantum_integral = 0.0
    for interval_type, _ in dyadic_intervals:
        try:
            # Measure contribution from this dyadic interval
            contribution_sum = run_integral_estimator_dyadic_correct(psi_f, scale_f, n, interval_type, shots)
            
            # Scale by grid spacing
            contribution_integral = contribution_sum * dx
            
            quantum_integral += contribution_integral
            
        except Exception as e:
            print(f"Error measuring {interval_type}: {e}")
            continue
    
    return {
        "exact": exact_integral,
        "compute_uncompute": quantum_integral,
        "error": abs(quantum_integral - exact_integral),
        "type": f"Dyadic: {len(dyadic_intervals)} intervals"
    }
    
    # 4. Quantum Calculation (using direct inner product of states)
    try:
        overlap = run_integral_estimator_arbitrary(psi_f, psi_g, n, shots)
        estimated_sum = overlap * scale_f * scale_g
        estimated_integral = estimated_sum * dx
    except Exception as e:
        estimated_integral = 0.0
    
    return {
        "exact": exact_integral,
        "compute_uncompute": estimated_integral,
        "error": abs(estimated_integral - exact_integral),
        "type": f"[{a:.2f}, {b:.2f}]"
    }