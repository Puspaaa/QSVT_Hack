import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from solvers import cvx_poly_coef, Angles_Fixed
from quantum import QSVT_circuit_universal, Advection_Gate

def exact_solution_fourier(u0_vals, t_phys, nu, c):
    N = len(u0_vals)
    ak = np.fft.fft(u0_vals)
    k_vals = np.fft.fftfreq(N, d=1/N)
    decay = np.exp(-4 * (np.pi**2) * nu * (k_vals**2) * t_phys)
    phase = np.exp(-2j * np.pi * c * k_vals * t_phys)
    ak_new = ak * decay * phase
    u_exact = np.fft.ifft(ak_new).real
    return u_exact / np.linalg.norm(u_exact)

def get_classical_matrix(N, nu, c):
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

def run_split_step_sim_exponential(n, nu, c, time_steps, u0_func, shots=100000):
    N = 2**n
    results = {}
    x_grid = np.linspace(0, 1, N, endpoint=False)
    u0_vals = u0_func(x_grid)
    state_vector = u0_vals / np.linalg.norm(u0_vals)
    dx = 1/N
    dt = 0.9 * dx**2 / (2 * nu)
    
    for t in time_steps:
        if t == 0:
            p0 = u0_vals**2
            results[0] = (x_grid, p0/np.sum(p0))
            continue
            
        deg = int(t + 8) 
        if deg % 2 != t % 2: deg += 1
        
        try:
            coef = cvx_poly_coef(lambda x: np.exp(t * (x - 1)), [0, 1], deg, epsil=1e-5)
            phi_seq = Angles_Fixed(coef)
        except Exception as e:
            print(f"Solver Error: {e}")
            continue
        
        qc = QSVT_circuit_universal(phi_seq, n, nu, init_state=state_vector, measurement=False)
        physical_time = t * dt
        data_reg = qc.qregs[2]
        qc.append(Advection_Gate(n, c, physical_time), data_reg)
        qc.measure(qc.qregs[0], qc.cregs[0])
        qc.measure(qc.qregs[1], qc.cregs[1])
        qc.measure(qc.qregs[2], qc.cregs[2])
        
        backend = AerSimulator()
        tqc = transpile(qc, backend, optimization_level=0)
        counts = backend.run(tqc, shots=shots).result().get_counts()
        
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
        
        if total_valid > 0: results[t] = (x_grid, prob_dist / total_valid)
        else: results[t] = (x_grid, np.zeros(N))
            
    return results