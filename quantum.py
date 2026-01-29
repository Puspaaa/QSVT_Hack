import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate, DiagonalGate

# ==============================================================================
# 1. QUANTUM GATES (Diffusion & Advection)
# ==============================================================================

class Shift_gate(QuantumCircuit):
    """Efficient cyclic shift using QFT."""
    def __init__(self, n, inverse=False):
        super().__init__(n, name="Shift" if not inverse else "Shift_dag")
        self.append(QFTGate(n), range(n))

        # Apply Phase Gradients
        for i in range(n):
            phase = 2 * np.pi * (2**i) / (2**n)
            if inverse: phase *= -1
            self.p(phase, i)

        self.append(QFTGate(n).inverse(), range(n))

def Block_encoding_diffusion(n, nu):
    """
    Constructs the Pure Diffusion Operator A (Hermitian).
    We use this for the QSVT part.
    """
    ancilla = QuantumRegister(2, 'anc')
    data = QuantumRegister(n, 'data')
    qc = QuantumCircuit(ancilla, data)

    # Physics Constants (Pure Diffusion)
    dx = 1/2**n
    dt = 0.9*dx**2/(2*nu)
    a_val = 1 - 2*dt*nu/(dx**2)
    theta = 2 * np.arccos(np.sqrt(a_val))

    # Prepare LCU Coefficients
    qc.ry(theta, ancilla[1])
    qc.ch(ancilla[1], ancilla[0])

    # Select Operations
    S = Shift_gate(n).to_gate()
    S_dag = S.inverse()

    # Apply S_dag if 10, S if 11
    qc.append(S_dag.control(2, ctrl_state='10'), ancilla[:] + data[:])
    qc.append(S.control(2, ctrl_state='11'), ancilla[:] + data[:])

    # Un-prepare
    qc.ch(ancilla[1], ancilla[0])
    qc.ry(-theta, ancilla[1])
    return qc

def Advection_Gate(n, c, physical_time):
    """
    Implements exact advection e^{-c t d/dx} using QFT.
    This handles the MOVEMENT of the wave.
    """
    qc = QuantumCircuit(n, name="Advection")

    # 1. Move to Fourier Space
    qc.append(QFTGate(n), range(n))

    # 2. Apply Momentum Phases (Shift Theorem)
    N = 2**n
    shift_distance = c * physical_time
    diagonals = []

    for j in range(N):
        # Map index j to signed frequency k
        # Standard FFT ordering: 0..N/2-1 are pos, N/2..N-1 are neg
        k = j if j < N/2 else j - N

        # Phase shift = -2*pi * k * shift_distance
        # (Assuming periodic boundary on [0,1])
        angle = 2 * np.pi * k * shift_distance
        diagonals.append(np.exp(1j * angle))

    qc.append(DiagonalGate(diagonals), range(n))

    # 3. Move back to Position Space
    qc.append(QFTGate(n).inverse(), range(n))
    return qc.to_gate(label=f"Adv(t={physical_time:.3f})")

def QSVT_circuit_universal(phi_seq, n, nu, init_state=None, measurement=True):
    """
    Standard QSVT Circuit.
    Added 'measurement' flag to allow appending gates later (like Advection).
    """
    sig = QuantumRegister(1, 'sig')
    anc = QuantumRegister(2, 'anc')
    dat = QuantumRegister(n, 'dat')

    # We always create registers, but only measure if requested
    c_sig = ClassicalRegister(1, 'm_sig')
    c_anc = ClassicalRegister(2, 'm_anc')
    c_dat = ClassicalRegister(n, 'm_dat')

    qc = QuantumCircuit(sig, anc, dat, c_sig, c_anc, c_dat)

    # Use Pure Diffusion Block Encoding
    U_gate = Block_encoding_diffusion(n, nu).to_gate(label="U")
    U_dag = U_gate.inverse()
    U_dag.label = "U†"

    if init_state is not None: qc.initialize(init_state, dat)
    qc.barrier()

    # 1. First Rotation (Global)
    qc.h(sig)
    qc.rz(2 * phi_seq[0], sig)
    qc.barrier()

    # 2. Loop (U + Reflection)
    for i in range(1, len(phi_seq)):
        gate = U_gate if i % 2 == 1 else U_dag
        qc.append(gate, anc[:] + dat[:])

        qc.x(anc)
        qc.ccx(anc[0], anc[1], sig)
        qc.rz(2 * phi_seq[i], sig)
        qc.ccx(anc[0], anc[1], sig)
        qc.x(anc)
        qc.barrier()

    # 3. Measure (Only if requested)
    if measurement:
        qc.h(sig)
        qc.measure(sig, c_sig)
        qc.measure(anc, c_anc)
        qc.measure(dat, c_dat)
    else:
        # Just the final basis change for Signal
        qc.h(sig)

    return qc