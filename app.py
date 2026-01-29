import streamlit as st

st.set_page_config(page_title="QSVT PDE Solver", layout="wide")

st.title("Quantum Singular Value Transformation for PDE Solving")

st.markdown(r"""
## Motivation

Advection–diffusion models transport and spreading. On fine grids, classical solvers scale at least linearly in the number of grid points $N$. QSVT enables polynomial transformations of structured operators that can reduce dependence on $N$ for certain tasks.

### Visual Summary

**State encoding**
$$N=2^n \Rightarrow n\;\text{qubits}$$

**Operator pipeline**
$$u_0 \xrightarrow{\text{Block-encode }A} \text{QSVT } P(A) \xrightarrow{\text{Advection via QFT}} u(t)$$

**Classical vs Quantum (qualitative)**

| Method | Cost per step (qualitative) | Notes |
|---|---|---|
| Classical grid | $\Omega(N)$ | Direct stencil updates |
| QSVT + block-encoding | polylog$(N)$ in structured cases | Depends on degree $d$ and postselection |

## App Structure

Use the sidebar to open:

- 1D advection–diffusion (PDE solver)
- 2D advection–diffusion (PDE solver)
- Problem 2: Integral Estimation (quantum measurement)

Each page follows: theory → configuration → computation → results.
""")

st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Quantum Algorithm", "QSVT")
with col2:
    st.metric("Framework", "Qiskit 1.x")
with col3:
    st.metric("Backend", "Aer Simulator")