# 🌊 Quantum Advection-Diffusion Wind Tunnel

A Streamlit application that simulates the advection-diffusion equation using quantum computing techniques, comparing three different solution methods: Exact Fourier, Classical Finite Difference, and Quantum Split-Step with QSVT (Quantum Singular Value Transformation).

## Overview

This application demonstrates how quantum computing can be applied to solve partial differential equations (PDEs) in computational fluid dynamics. The advection-diffusion equation models how quantities like heat, concentration, or pollutants spread and move through space:

```
∂u/∂t = ν∂²u/∂x² - v∂u/∂x
```

where:
- **ν** (nu) is the viscosity/diffusion coefficient
- **v** is the advection velocity  
- **u(x,t)** is the quantity being transported

## Features

### 📊 Three Solution Methods
1. **Exact Fourier Solution** - Reference solution using Fourier transforms
2. **Classical Finite Difference** - Traditional numerical method with Crank-Nicolson scheme
3. **Quantum Split-Step** - Quantum circuit-based simulation using operator splitting

### 🔬 Quantum Circuit Implementation
- **Diffusion Block Encoding** - QFT-based implementation of the diffusion operator
- **Advection Gate** - Unitary shift operator for transport
- **QSVT Enhancement** - Polynomial transformation for improved accuracy

### 🎯 Interactive Features
- Real-time parameter adjustment (qubits, viscosity, velocity, time)
- Visual comparison plots with error analysis
- Circuit depth and gate count statistics
- Comprehensive mathematical background

## Project Structure

```
QSVT_Hack/
├── app.py           # Main Streamlit UI with tabs and controls
├── simulation.py    # Comparison logic and plotting functions
├── quantum.py       # Qiskit quantum circuit implementations
├── solvers.py       # CVXPY optimization for polynomial coefficients
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Puspaaa/QSVT_Hack.git
cd QSVT_Hack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Adjust Parameters** (Left Sidebar):
   - **Number of Qubits** (3-8): Controls grid resolution (Grid points = 2^n_qubits)
   - **Viscosity** (0.001-0.5): Diffusion coefficient
   - **Velocity** (-2.0 to 2.0): Advection velocity
   - **Final Time** (0.1-5.0): Simulation time

2. **Run Simulation** (Results Tab):
   - Click "🚀 Run Simulation" to execute all three methods
   - View comparison plots and error metrics
   - See performance statistics

3. **Explore Circuit Details** (Circuit Details Tab):
   - Review quantum circuit architecture
   - Check circuit depth and gate counts
   - Generate combined circuit visualization

4. **Learn the Math** (Mathematical Background Tab):
   - Understand the underlying equations
   - Compare solution methods
   - Explore QSVT polynomial approximations

## Technical Details

### Quantum Circuit Components

**Diffusion Block Encoding:**
```
QFT → Phase Rotations (exp(-ν*dt*k²)) → Inverse QFT
```

**Advection Gate:**
```
QFT → Phase Rotations (exp(-i*v*dt*k)) → Inverse QFT
```

**Split-Step Evolution:**
```
U(Δt) = exp(-ν*Δt/2 * ∂²/∂x²) · exp(-v*Δt * ∂/∂x) · exp(-ν*Δt/2 * ∂²/∂x²)
```

### Dependencies

- **streamlit**: Web application framework
- **qiskit**: Quantum computing SDK
- **qiskit-aer**: Quantum circuit simulator
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing
- **cvxpy**: Convex optimization for QSVT coefficients

## Examples

### Example 1: Low Viscosity, High Velocity
```python
qubits = 5
viscosity = 0.01
velocity = 1.5
time = 1.0
```
Result: Dominant advection with minimal spreading

### Example 2: High Viscosity, Low Velocity
```python
qubits = 5
viscosity = 0.2
velocity = 0.2
time = 2.0
```
Result: Dominant diffusion with significant spreading

## Performance Notes

- **Grid Resolution**: Higher qubit counts provide finer resolution but increase computation time
- **Quantum Simulation**: Currently uses classical simulation of quantum circuits (Qiskit Aer)
- **Real Quantum Hardware**: Code is compatible with IBM Quantum backends (requires additional setup)

## Future Enhancements

- [ ] Integration with real quantum hardware (IBMQ)
- [ ] Support for 2D and 3D advection-diffusion
- [ ] Adaptive time-stepping algorithms
- [ ] Error mitigation techniques
- [ ] Advanced QSVT polynomial optimization
- [ ] Parallel quantum circuit execution

## References

### Scientific Background
- **Advection-Diffusion Equation**: Classical PDE in fluid dynamics
- **Operator Splitting**: Strang splitting for time integration
- **QSVT**: Quantum Singular Value Transformation framework

### Quantum Computing
- **Qiskit Documentation**: https://qiskit.org/documentation/
- **QSVT Papers**: Martyn et al., "Grand Unification of Quantum Algorithms"
- **Block Encoding**: Techniques for encoding matrices as unitaries

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with Streamlit, Qiskit, and NumPy
- Quantum algorithms based on QSVT framework
- Inspired by quantum PDE solver research

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a demonstration and educational tool. For production quantum PDE solving, additional optimization and error correction would be required.