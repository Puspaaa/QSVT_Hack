import numpy as np

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
