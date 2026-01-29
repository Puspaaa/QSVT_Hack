"""
Quantum circuit implementations for advection-diffusion simulation using QSVT.
Includes diffusion block encoding, advection gate, and QSVT circuit construction.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from typing import List, Tuple, Optional
from solvers import compute_qsvt_polynomial_coefficients, optimize_phase_factors


def create_diffusion_block_encoding(
    n_qubits: int,
    viscosity: float,
    dt: float = 0.1
) -> QuantumCircuit:
    """
    Create a block-encoded quantum circuit for the diffusion operator.
    
    The diffusion operator is exp(-ν * dt * ∂²/∂x²), which is diagonal
    in Fourier space with eigenvalues exp(-ν * dt * k²).
    
    Args:
        n_qubits: Number of qubits for spatial discretization
        viscosity: Diffusion coefficient ν
        dt: Time step
    
    Returns:
        Quantum circuit implementing block-encoded diffusion
    """
    qc = QuantumCircuit(n_qubits, name='Diffusion')
    
    # Apply QFT to go to Fourier space
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))
    
    # Apply phase rotations for diffusion
    # In Fourier space: exp(-ν * dt * k²)
    N = 2 ** n_qubits
    for k_idx in range(N):
        # Wave number
        k = 2 * np.pi * (k_idx if k_idx < N//2 else k_idx - N) / N
        
        # Phase from diffusion: -ν * dt * k²
        phase = -viscosity * dt * k**2
        
        # Apply controlled phase rotation
        # We need to apply this phase when the state is |k_idx⟩
        if abs(phase) > 1e-10:  # Only if significant
            _apply_controlled_phase_to_basis_state(qc, k_idx, phase, n_qubits)
    
    # Apply inverse QFT to return to position space
    qc.append(QFT(n_qubits, do_swaps=True).inverse(), range(n_qubits))
    
    return qc


def create_advection_gate(
    n_qubits: int,
    velocity: float,
    dt: float = 0.1
) -> QuantumCircuit:
    """
    Create a quantum gate for the advection (shift) operator.
    
    The advection operator is exp(-v * dt * ∂/∂x), which in Fourier space
    becomes exp(-i * v * dt * k), representing a translation.
    
    Args:
        n_qubits: Number of qubits
        velocity: Advection velocity v
        dt: Time step
    
    Returns:
        Quantum circuit implementing advection
    """
    qc = QuantumCircuit(n_qubits, name='Advection')
    
    # Apply QFT to go to Fourier space
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))
    
    # Apply phase rotations for advection
    # In Fourier space: exp(-i * v * dt * k)
    N = 2 ** n_qubits
    for k_idx in range(N):
        # Wave number
        k = 2 * np.pi * (k_idx if k_idx < N//2 else k_idx - N) / N
        
        # Phase from advection: -v * dt * k
        phase = -velocity * dt * k
        
        # Apply phase rotation
        if abs(phase) > 1e-10:
            _apply_controlled_phase_to_basis_state(qc, k_idx, phase, n_qubits)
    
    # Apply inverse QFT to return to position space
    qc.append(QFT(n_qubits, do_swaps=True).inverse(), range(n_qubits))
    
    return qc


def _apply_controlled_phase_to_basis_state(
    qc: QuantumCircuit,
    basis_state: int,
    phase: float,
    n_qubits: int
) -> None:
    """
    Apply a phase to a specific computational basis state.
    
    Uses multi-controlled phase gates to target a specific basis state |basis_state⟩.
    
    Args:
        qc: Quantum circuit to modify
        basis_state: Index of basis state to apply phase to
        phase: Phase angle in radians
        n_qubits: Number of qubits
    """
    # Convert basis state to binary
    binary = format(basis_state, f'0{n_qubits}b')
    
    # Apply X gates to flip qubits where bit is 0
    for i, bit in enumerate(binary):
        if bit == '0':
            qc.x(i)
    
    # Apply multi-controlled phase gate
    if n_qubits == 1:
        qc.p(phase, 0)
    elif n_qubits == 2:
        qc.cp(phase, 0, 1)
    else:
        # For more qubits, use multi-controlled phase
        # Simplification: apply phase to each qubit
        control_qubits = list(range(n_qubits - 1))
        target_qubit = n_qubits - 1
        qc.mcp(phase, control_qubits, target_qubit)
    
    # Undo X gates
    for i, bit in enumerate(binary):
        if bit == '0':
            qc.x(i)


def create_qsvt_circuit(
    n_qubits: int,
    block_encoding: QuantumCircuit,
    phase_factors: np.ndarray
) -> QuantumCircuit:
    """
    Create a QSVT (Quantum Singular Value Transformation) circuit.
    
    QSVT applies a polynomial transformation to the singular values of
    a block-encoded matrix using a sequence of phase factors.
    
    Args:
        n_qubits: Number of qubits
        block_encoding: Block-encoded operator circuit
        phase_factors: Array of phase factors for QSVT
    
    Returns:
        QSVT quantum circuit
    """
    # Create circuit with ancilla qubit for block encoding
    qreg = QuantumRegister(n_qubits, 'q')
    anc = QuantumRegister(1, 'anc')
    qc = QuantumCircuit(qreg, anc, name='QSVT')
    
    n_phases = len(phase_factors)
    
    # QSVT sequence: alternate between signal operator and projector-controlled phases
    for i in range(n_phases):
        # Apply signal operator (block encoding)
        if i > 0:
            # Controlled application on ancilla
            controlled_block = block_encoding.control(1)
            qc.append(controlled_block, [anc[0]] + list(qreg))
        
        # Apply projector-controlled phase
        # This implements: e^{iφ_i |0⟩⟨0|}
        qc.p(phase_factors[i], anc[0])
        
        # Apply X gate on ancilla to flip projector
        qc.x(anc[0])
        qc.p(-phase_factors[i], anc[0])
        qc.x(anc[0])
    
    return qc


def create_combined_advection_diffusion_circuit(
    n_qubits: int,
    viscosity: float,
    velocity: float,
    n_steps: int = 1,
    dt: float = 0.1,
    use_qsvt: bool = False
) -> QuantumCircuit:
    """
    Create a combined circuit for advection-diffusion using split-step method.
    
    Args:
        n_qubits: Number of qubits
        viscosity: Diffusion coefficient
        velocity: Advection velocity
        n_steps: Number of time steps to simulate
        dt: Time step size
        use_qsvt: Whether to use QSVT enhancement
    
    Returns:
        Combined quantum circuit
    """
    qc = QuantumCircuit(n_qubits, name='AdvectionDiffusion')
    
    # Create component circuits
    diffusion_circ = create_diffusion_block_encoding(n_qubits, viscosity, dt)
    advection_circ = create_advection_gate(n_qubits, velocity, dt)
    
    # Split-step evolution: alternate diffusion and advection
    for step in range(n_steps):
        # Half step diffusion
        qc.append(diffusion_circ, range(n_qubits))
        
        # Full step advection
        qc.append(advection_circ, range(n_qubits))
        
        # Half step diffusion
        qc.append(diffusion_circ, range(n_qubits))
    
    if use_qsvt:
        # Optionally wrap in QSVT for enhanced precision
        # Compute phase factors
        degree = min(10, 2 * n_qubits)
        phase_factors = optimize_phase_factors(
            n_phases=degree,
            target_singular_values=np.linspace(-1, 1, 50)
        )
        
        # Create QSVT circuit
        qsvt_circ = create_qsvt_circuit(n_qubits, qc, phase_factors)
        return qsvt_circ
    
    return qc


def simulate_quantum_advection_diffusion(
    n_qubits: int,
    initial_state: np.ndarray,
    viscosity: float,
    velocity: float,
    time_points: np.ndarray
) -> np.ndarray:
    """
    Simulate advection-diffusion using quantum circuits.
    
    Args:
        n_qubits: Number of qubits
        initial_state: Initial state vector (length 2^n_qubits)
        viscosity: Diffusion coefficient
        velocity: Advection velocity
        time_points: Array of time points to evaluate
    
    Returns:
        Array of states at each time point (shape: [len(time_points), 2^n_qubits])
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    
    N = 2 ** n_qubits
    results = np.zeros((len(time_points), N), dtype=complex)
    results[0] = initial_state
    
    # Simulate evolution
    for i, t in enumerate(time_points[1:], start=1):
        dt = t - time_points[i-1]
        
        # Create circuit for this time step
        qc = create_combined_advection_diffusion_circuit(
            n_qubits, viscosity, velocity, n_steps=1, dt=dt
        )
        
        # Initialize with previous state
        state = Statevector(results[i-1])
        state = state.evolve(qc)
        
        # Store result
        results[i] = state.data
    
    return np.abs(results) ** 2  # Return probabilities
