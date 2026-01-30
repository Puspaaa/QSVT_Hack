# Quantum Singular Value Transformation for Scientific Computing

A Streamlit application demonstrating QSVT for **PDE solving** and **numerical integration** — built for QHack 2026.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 5-Minute Presentation Flow

| Time | Demo | Key Takeaway |
|------|------|--------------|
| **0:00-0:30** | Landing Page | The challenge: exponential state space; the insight: QSVT polynomial transformations |
| **0:30-2:00** | 1D PDE Solver | Watch quantum simulate advection-diffusion; compare to classical |
| **2:00-4:30** | Problem 2: Integration | Three methods, arbitrary intervals, run comparison |
| **4:30-5:00** | Q&A | Use "Technical Deep Dive" expanders for details |

## Project Structure

```
QSVT_Hack/
├── app.py                    # Landing page + QSVT overview
├── pages/
│   ├── 1_1D_Simulation.py    # 1D advection-diffusion solver
│   ├── 2_2D_Simulation.py    # 2D extension
│   └── 3_Problem_2_Integrals.py  # Quantum integration (3 methods)
├── quantum.py                # Quantum circuits (QSVT, block encoding)
├── measurements.py           # Integration measurement routines
├── simulation.py             # PDE simulation logic
├── solvers.py                # Polynomial/angle computation
└── requirements.txt          # Dependencies
```

## Demo 1: Advection-Diffusion PDE

Solves the 1D/2D advection-diffusion equation:

```
∂u/∂t = ν∇²u - c·∇u
```

**Quantum approach:**
1. **Block-encode** the Laplacian as a unitary subblock
2. **QSVT** applies the time evolution polynomial P(A) = e^{(A-I)t}
3. **Measure** to extract the evolved state

## Demo 2: Quantum Integration (Problem 2)

Computes ∫_a^b f(x) dx using three methods:

| Method | Intervals | Complexity | Error |
|--------|-----------|------------|-------|
| **Compute-Uncompute** | Special (half-intervals) | O(n) | < 1% |
| **Arithmetic/Comparison** | ANY [a, b] | O(M·n) | 3-15% |
| **QSVT Parity** | ANY [a, b] | O(d·n) | 5-30% |

**Key insight:** Integration is an inner product → quantum overlap estimation!

## Technical Highlights

- **Qiskit 1.x** with AerSimulator backend
- **QSVT implementation** with Chebyshev polynomial approximation
- **Robust angle finding** via CVXPY optimization
- **MCX comparator** for arbitrary interval marking

## References

- [Grand Unification of Quantum Algorithms (Martyn et al.)](https://arxiv.org/abs/2105.02859)
- [QSVT Tutorial (Gilyen et al.)](https://arxiv.org/abs/1806.01838)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## License

MIT License - QHack 2026

## Acknowledgments

- Built with Streamlit, Qiskit, and NumPy
- Quantum algorithms based on QSVT framework
- Inspired by quantum PDE solver research

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is a demonstration and educational tool. For production quantum PDE solving, additional optimization and error correction would be required.