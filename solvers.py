import numpy as np
import cvxpy as cp
from numpy.polynomial.chebyshev import Chebyshev, chebpts1
from pyqsp import angle_sequence

def cvx_poly_coef(targ_f, interval, deg, epsil=1e-6, npts=2000):
    """Weighted Least Squares Solver for Chebyshev Coefficients."""
    parity = deg % 2
    a, b = interval
    xpts = 0.5 * (a + b) + 0.5 * (b - a) * chebpts1(2 * npts)
    npts = len(xpts)

    n_coef = int(np.floor(deg/2)+1)
    Ax = np.zeros((npts, n_coef))
    for k in range(1, n_coef+1):
        Tcheb = Chebyshev.basis(2*(k-1)+parity)
        Ax[:, k-1] = Tcheb(xpts)

    coef = cp.Variable(n_coef)
    y = cp.Variable(npts)

    # Weight errors in valleys (value near 0) 100x more than peaks
    targ_vals = targ_f(xpts)
    weights = np.ones(npts)
    weights[np.abs(targ_vals) < 0.1] = 100.0

    constraints = [y == Ax @ coef, y >= -(1-epsil), y <= (1-epsil)]
    error = cp.multiply(weights, (y - targ_vals))

    prob = cp.Problem(cp.Minimize(cp.norm(error, 2)), constraints)
    prob.solve()

    c = np.array(coef.value)
    coef_full = np.zeros(deg + 1)
    if parity == 0: coef_full[::2] = c
    else: coef_full[1::2] = c
    return coef_full

def Angles_Fixed(coef, tolerance=1e-5):
    """Reliable Angle Finder (Returns d+1 angles)."""
    (phiset, _, _) = angle_sequence.QuantumSignalProcessingPhases(
        poly=coef, eps=1e-6, suc=1-1e-6, signal_operator="Wx",
        tolerance=tolerance, method="sym_qsp", chebyshev_basis=True
    )
    Phi = np.array(phiset)
    # Convert QSP (Rx) to QSVT (Rz Reflection)
    if len(Phi) > 2: Phi[1:-1] -= np.pi/2
    return Phi