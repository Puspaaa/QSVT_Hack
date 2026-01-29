import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate, DiagonalGate

class Shift_gate(QuantumCircuit):
    """Efficient cyclic shift using QFT."""
    def __init__(self, n, inverse=False):
        super().__init__(n, name="Shift" if not inverse else "Shift_dag")
        self.append(QFTGate(n), range(n))
        for i in range(n):
            phase = 2 * np.pi * (2**i) / (2**n)
            if inverse: phase *= -1
            self.p(phase, i)
        self.append(QFTGate(n).inverse(), range(n))

def Block_encoding_diffusion(n, nu):
    """LCU block-encoding of diffusion operator."""
    ancilla = QuantumRegister(2, 'anc')
    data = QuantumRegister(n, 'data')
    qc = QuantumCircuit(ancilla, data)
    dx = 1/2**n
    dt = 0.9*dx**2/(2*nu)
    a_val = 1 - 2*dt*nu/(dx**2)
    theta = 2 * np.arccos(np.sqrt(a_val))
    qc.ry(theta, ancilla[1])
    qc.ch(ancilla[1], ancilla[0])
    S = Shift_gate(n).to_gate()
    S_dag = S.inverse()
    qc.append(S_dag.control(2, ctrl_state='10'), ancilla[:] + data[:])
    qc.append(S.control(2, ctrl_state='11'), ancilla[:] + data[:])
    qc.ch(ancilla[1], ancilla[0])
    qc.ry(-theta, ancilla[1])
    return qc

def Advection_Gate(n, c, physical_time):
    """QFT-based advection operator: e^{-c t d/dx}."""
    qc = QuantumCircuit(n, name="Advection")
    qc.append(QFTGate(n), range(n))
    N = 2**n
    shift_distance = c * physical_time
    diagonals = []
    for j in range(N):
        k = j if j < N/2 else j - N
        angle = 2 * np.pi * k * shift_distance
        diagonals.append(np.exp(1j * angle))
    qc.append(DiagonalGate(diagonals), range(n))
    qc.append(QFTGate(n).inverse(), range(n))
    return qc.to_gate(label=f"Adv(t={physical_time:.3f})")

def QSVT_circuit_universal(phi_seq, n, nu, init_state=None, measurement=True):
    """QSVT circuit with optional measurement."""
    sig = QuantumRegister(1, 'sig')
    anc = QuantumRegister(2, 'anc')
    dat = QuantumRegister(n, 'dat')
    c_sig = ClassicalRegister(1, 'm_sig')
    c_anc = ClassicalRegister(2, 'm_anc')
    c_dat = ClassicalRegister(n, 'm_dat')
    qc = QuantumCircuit(sig, anc, dat, c_sig, c_anc, c_dat)
    U_gate = Block_encoding_diffusion(n, nu).to_gate(label="U")
    U_dag = U_gate.inverse()
    U_dag.label = "U†"
    if init_state is not None: qc.initialize(init_state, dat)
    qc.barrier()
    qc.h(sig)
    qc.rz(2 * phi_seq[0], sig)
    qc.barrier()
    for i in range(1, len(phi_seq)):
        gate = U_gate if i % 2 == 1 else U_dag
        qc.append(gate, anc[:] + dat[:])
        qc.x(anc)
        qc.ccx(anc[0], anc[1], sig)
        qc.rz(2 * phi_seq[i], sig)
        qc.ccx(anc[0], anc[1], sig)
        qc.x(anc)
        qc.barrier()
    qc.h(sig)
    if measurement:
        qc.measure(sig, c_sig)
        qc.measure(anc, c_anc)
        qc.measure(dat, c_dat)
    return qc

# ==============================================================================
# 2D QUANTUM CIRCUITS FOR 2D DIFFUSION-ADVECTION
# ==============================================================================

def Shift_gate_2d(nx, ny, axis=0, inverse=False):
    """Cyclic shift in 2D grid along specified axis."""
    n_total = nx + ny
    qc = QuantumCircuit(n_total, name=f"Shift_2d_ax{axis}" + ("_dag" if inverse else ""))
    
    # For 2D grid flattened as idx = x + y*nx:
    # Shift along x (axis=0): increment x index (wraps at nx)
    # Shift along y (axis=1): increment y index (wraps at ny)
    if axis == 0:  # Shift along x-axis
        qc.append(QFTGate(nx), range(nx))
        for i in range(nx):
            phase = 2 * np.pi * (2**i) / (2**nx)
            if inverse: phase *= -1
            qc.p(phase, i)
        qc.append(QFTGate(nx).inverse(), range(nx))
    else:  # Shift along y-axis
        qc.append(QFTGate(ny), range(nx, nx+ny))
        for i in range(ny):
            phase = 2 * np.pi * (2**i) / (2**ny)
            if inverse: phase *= -1
            qc.p(phase, nx + i)
        qc.append(QFTGate(ny).inverse(), range(nx, nx+ny))
    
    return qc

def Block_encoding_diffusion_2d(nx, ny, nu):
    """
    2D Block Encoding for Laplacian operator.
    Uses 4-neighbor stencil (5-point: center + 4 neighbors).
    Grid indexed as: flat_idx = x + y*nx where x in [0,nx), y in [0,ny)
    """
    ancilla = QuantumRegister(2, 'anc')
    x_reg = QuantumRegister(nx, 'x')
    y_reg = QuantumRegister(ny, 'y')
    qc = QuantumCircuit(ancilla, x_reg, y_reg)
    
    # Physics: 2D diffusion on [0,1) x [0,1)
    dx = 1 / nx
    dy = 1 / ny
    dt = 0.9 * min(dx**2, dy**2) / (4 * nu)  # 4 neighbors in 2D
    
    # Laplacian coefficient (2D: -4 on center, +1 on each neighbor)
    a_center = 1 - 4 * dt * nu / (dx**2)
    theta = 2 * np.arccos(np.sqrt(a_center))
    
    # Prepare 5 unitaries: I, S_x, S_x†, S_y, S_y†
    qc.ry(theta, ancilla[1])
    qc.ch(ancilla[1], ancilla[0])
    
    # Define shift gates
    S_x = Shift_gate_2d(nx, ny, axis=0).to_gate()
    S_x_dag = S_x.inverse()
    S_y = Shift_gate_2d(nx, ny, axis=1).to_gate()
    S_y_dag = S_y.inverse()
    
    # Apply shifts based on ancilla state (5 states from 2 ancilla is tricky)
    # State |00⟩: Identity (do nothing)
    # State |01⟩: S_x (right)
    # State |10⟩: S_x† (left)
    # State |11⟩: S_y (up) then S_y† (down) via sequential control
    
    qc.append(S_x.control(2, ctrl_state='01'), ancilla[:] + x_reg[:] + y_reg[:])
    qc.append(S_x_dag.control(2, ctrl_state='10'), ancilla[:] + x_reg[:] + y_reg[:])
    qc.append(S_y.control(2, ctrl_state='11'), ancilla[:] + x_reg[:] + y_reg[:])
    
    # Un-prepare
    qc.ch(ancilla[1], ancilla[0])
    qc.ry(-theta, ancilla[1])
    
    return qc

def Advection_Gate_2d(nx, ny, c_x, c_y, physical_time):
    """
    2D Advection: e^{-c_x t ∂/∂x - c_y t ∂/∂y}
    Applied as independent QFT phase shifts on x and y axes.
    """
    n_total = nx + ny
    qc = QuantumCircuit(n_total, name=f"Adv2d(t={physical_time:.3f})")
    
    x_reg = list(range(nx))
    y_reg = list(range(nx, nx + ny))
    
    # === X-Advection ===
    qc.append(QFTGate(nx), x_reg)
    shift_x = c_x * physical_time
    for j in range(nx):
        k = j if j < nx/2 else j - nx
        angle_x = 2 * np.pi * k * shift_x
        qc.append(DiagonalGate([1.0, np.exp(1j * angle_x)]), [x_reg[j]])
    qc.append(QFTGate(nx).inverse(), x_reg)
    
    # === Y-Advection ===
    qc.append(QFTGate(ny), y_reg)
    shift_y = c_y * physical_time
    for j in range(ny):
        k = j if j < ny/2 else j - ny
        angle_y = 2 * np.pi * k * shift_y
        qc.append(DiagonalGate([1.0, np.exp(1j * angle_y)]), [y_reg[j]])
    qc.append(QFTGate(ny).inverse(), y_reg)
    
    return qc.to_gate()

def QSVT_circuit_2d(phi_seq, nx, ny, nu, init_state=None, measurement=True):
    """
    2D QSVT Circuit for 2D Laplacian.
    Similar structure to 1D but with 2D block encoding.
    """
    sig = QuantumRegister(1, 'sig')
    anc = QuantumRegister(2, 'anc')
    x_reg = QuantumRegister(nx, 'x')
    y_reg = QuantumRegister(ny, 'y')
    
    c_sig = ClassicalRegister(1, 'm_sig')
    c_anc = ClassicalRegister(2, 'm_anc')
    c_dat = ClassicalRegister(nx + ny, 'm_dat')
    
    qc = QuantumCircuit(sig, anc, x_reg, y_reg, c_sig, c_anc, c_dat)
    
    # 2D Block Encoding
    U_gate = Block_encoding_diffusion_2d(nx, ny, nu).to_gate(label="U_2d")
    U_dag = U_gate.inverse()
    U_dag.label = "U†_2d"
    
    if init_state is not None:
        qc.initialize(init_state, x_reg[:] + y_reg[:])
    qc.barrier()
    
    # First rotation
    qc.h(sig)
    qc.rz(2 * phi_seq[0], sig)
    qc.barrier()
    
    # QSVT loop
    for i in range(1, len(phi_seq)):
        gate = U_gate if i % 2 == 1 else U_dag
        qc.append(gate, anc[:] + x_reg[:] + y_reg[:])
        
        qc.x(anc)
        qc.ccx(anc[0], anc[1], sig)
        qc.rz(2 * phi_seq[i], sig)
        qc.ccx(anc[0], anc[1], sig)
        qc.x(anc)
        qc.barrier()
    
    # Measurements
    if measurement:
        qc.h(sig)
        qc.measure(sig, c_sig)
        qc.measure(anc, c_anc)
        qc.measure(x_reg, c_dat[:nx])
        qc.measure(y_reg, c_dat[nx:])
    else:
        qc.h(sig)
    
    return qc

    return qc