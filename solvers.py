import numpy as np
import cvxpy as cp
from math import exp, log
from numpy.polynomial.chebyshev import Chebyshev, chebpts1, chebfit, chebval
from pyqsp import angle_sequence


def paper_r_function(M, eps, max_iter=80):
    """Solve for r in (M/r)^r = eps with r > M.

    This helper mirrors Definition 5.1 in arXiv:2512.22163 and is used for
    practical degree estimates in the Chebyshev truncation formulas.
    """
    if M <= 0:
        raise ValueError("M must be positive")
    if not (0 < eps < 1):
        raise ValueError("eps must satisfy 0 < eps < 1")

    # g(r) = r*log(M/r) - log(eps), root sought for r > M.
    def g(r):
        return r * (log(M) - log(r)) - log(eps)

    lo = max(M * (1.0 + 1e-12), M + 1e-12)
    hi = max(2.0 * M, M + 1.0)
    while g(hi) > 0:
        hi *= 2.0
        if hi > 1e12:
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paper_chebyshev_truncation_orders(M1, M2, eps):
    """Return (R1, R2, degree) using Proposition 5.5-style formulas.

    The returned degree corresponds to the product structure used to approximate
    exp(-M1 x^2 + i M2 x) in the paper.
    """
    if not (M1 >= 0 and M2 >= 0):
        raise ValueError("M1 and M2 must be non-negative")
    if not (0 < eps < 1):
        raise ValueError("eps must satisfy 0 < eps < 1")

    # If M1 or M2 is zero, keep the corresponding truncation minimal.
    if M1 == 0:
        R1 = 0
    else:
        R1 = int(np.floor(paper_r_function(exp(M1 / 4.0), 5.0 * eps / 12.0)))

    if M2 == 0:
        R2 = 0
    else:
        R2 = int(np.floor(0.5 * paper_r_function(exp(M2 / 2.0), 5.0 * eps / 8.0)))

    degree = 2 * (R1 + R2)
    return R1, R2, degree

def robust_poly_coef(targ_f, interval, deg, epsil=1e-6, npts=2000):
    """Fit Chebyshev coefficients with a fast least-squares approach.

    Recommended for interactive demos and integration tasks where numerical
    robustness and speed are more important than strict boundedness guarantees.
    This does not explicitly enforce $|P(x)| <= 1$; validate bounds separately
    if required by QSP constraints.
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
    """Solve constrained Chebyshev fitting with explicit QSP-style bounds.

    Recommended for PDE/simulation paths where preserving the bounded
    polynomial constraint is important. This method is slower than
    ``robust_poly_coef`` but enforces ``|P(x)| <= 1 - epsil`` on sample points.
    """
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