import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate, DiagonalGate

class Shift_gate(QuantumCircuit):
    def __init__(self, n, inverse=False):
        super().__init__(n, name="Shift" if not inverse else "Shift_dag")
        self.append(QFTGate(n, do_swaps=False), range(n))
        for i in range(n):
            phase = 2 * np.pi * (2**i) / (2**n)
            if inverse: phase *= -1
            self.p(phase, i)
        self.append(QFTGate(n, do_swaps=False).inverse(), range(n))

def Block_encoding_diffusion(n, nu):
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
    qc = QuantumCircuit(n, name="Advection")
    qc.append(QFTGate(n, do_swaps=False), range(n))
    
    N = 2**n
    shift_distance = c * physical_time
    diagonals = []
    
    for j in range(N):
        k = j if j < N/2 else j - N
        angle = 2 * np.pi * k * shift_distance 
        diagonals.append(np.exp(1j * angle))
        
    qc.append(DiagonalGate(diagonals), range(n))
    qc.append(QFTGate(n, do_swaps=False).inverse(), range(n))
    return qc.to_gate(label=f"Adv(t={physical_time:.3f})")

def QSVT_circuit_universal(phi_seq, n, nu, init_state=None, measurement=True):
    sig = QuantumRegister(1, 'sig')
    anc = QuantumRegister(2, 'anc')
    dat = QuantumRegister(n, 'dat')
    c_sig = ClassicalRegister(1, 'm_sig')
    c_anc = ClassicalRegister(2, 'm_anc')
    c_dat = ClassicalRegister(n, 'm_dat')
    
    qc = QuantumCircuit(sig, anc, dat, c_sig, c_anc, c_dat)
    U_gate = Block_encoding_diffusion(n, nu).to_gate(label="U_Diff")
    U_dag = U_gate.inverse()
    
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

    if measurement:
        qc.h(sig)
        qc.measure(sig, c_sig)
        qc.measure(anc, c_anc)
        qc.measure(dat, c_dat)
    else:
        qc.h(sig)
    return qc