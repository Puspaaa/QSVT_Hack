import numpy as np
import cvxpy as cp
from numpy.polynomial.chebyshev import Chebyshev, chebpts1, chebfit, chebval
from pyqsp import angle_sequence

def robust_poly_coef(targ_f, interval, deg, epsil=1e-6, npts=2000):
    """Robust Chebyshev polynomial fitting using stable least-squares method.
    
    This implementation uses NumPy's robust chebfit which handles normalization correctly.
    Used for Problem 2 (Integral).
    """
    a, b = interval
    
    # Generate points in the original domain [a, b]
    xpts = 0.5 * (a + b) + 0.5 * (b - a) * chebpts1(2 * npts)
    
    # Normalize to [-1, 1] for Chebyshev basis
    xpts_normalized = (2*xpts - (a+b)) / (b-a)
    
    # Evaluate target function at points
    targ_vals = targ_f(xpts)
    
    # Use NumPy's robust least-squares Chebyshev fitting
    coef_full = chebfit(xpts_normalized, targ_vals, deg)
    
    return coef_full

def cvx_poly_coef(targ_f, interval, deg, epsil=1e-6, npts=2000):
    """Original CVXPY-based solver for Problem 1 (Simulation)."""
    a, b = interval
    
    # Generate Chebyshev points in [a, b]
    xpts = 0.5 * (a + b) + 0.5 * (b - a) * chebpts1(2 * npts)
    
    # Target values
    targ_vals = targ_f(xpts)
    
    # Coefficients handling (parity)
    parity = deg % 2
    n_coef = int(np.floor(deg/2)+1)
    
    # Basis matrix
    Ax = np.zeros((len(xpts), n_coef))
    for k in range(1, n_coef+1):
        # Note: We evaluate Chebyshev polynomials on xpts directly.
        # This assumes xpts is within [-1, 1].
        # For P1 (simulation), interval is [0, 1], so this is valid.
        Tcheb = Chebyshev.basis(2*(k-1)+parity)
        Ax[:, k-1] = Tcheb(xpts)
    
    # Optimization
    coef = cp.Variable(n_coef)
    y = cp.Variable(len(xpts))
    
    # Constraints: |P(x)| <= 1 - epsilon (for QSP stability)
    constraints = [y == Ax @ coef, cp.abs(y) <= 1 - epsil]
    
    # Objective: Minimize weighted error
    # Weights focus accuracy on small values (typical for these problems)
    weights = np.ones(len(xpts))
    weights[np.abs(targ_vals) < 0.1] = 10.0 # Heuristic weight
    
    obj = cp.Minimize(cp.norm(cp.multiply(weights, y - targ_vals), 2))
    
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-6)
    except:
        prob.solve(solver=cp.ECOS)
        
    c = np.array(coef.value)
    
    # Expand to full coefficient array
    coef_full = np.zeros(deg + 1)
    if parity == 0: 
        coef_full[::2] = c
    else: 
        coef_full[1::2] = c
        
    return coef_full

def Angles_Fixed(coef, tolerance=1e-5):
    """Compute QSP phases and convert to QSVT format."""
    (phiset, _, _) = angle_sequence.QuantumSignalProcessingPhases(
        poly=coef, eps=1e-6, suc=1-1e-6, signal_operator="Wx",
        tolerance=tolerance, method="sym_qsp", chebyshev_basis=True
    )
    Phi = np.array(phiset)
    if len(Phi) > 2: Phi[1:-1] -= np.pi/2
    return Phi