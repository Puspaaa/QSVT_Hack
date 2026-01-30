import streamlit as st

st.set_page_config(
    page_title="QSVT for Scientific Computing", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("Quantum Singular Value Transformation")
st.markdown("**Block-Encoded Polynomial Transformations for PDEs and Numerical Integration**")

st.markdown("---")

# Core Concept
st.header("Theoretical Foundation")

st.markdown(r"""
QSVT provides a unified framework for quantum algorithms by implementing **polynomial transformations** 
of block-encoded matrices. Given a unitary $U_A$ that block-encodes a matrix $A$ (i.e., $A$ appears as the 
top-left block of $U_A$), QSVT constructs a circuit that block-encodes $P(A)$ for any bounded polynomial $P$.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Block Encoding Definition**")
    st.latex(r"U_A = \begin{pmatrix} A/\alpha & \cdot \\ \cdot & \cdot \end{pmatrix}, \quad \|A\| \leq \alpha")
    st.markdown(r"""
    The ancilla qubits project onto the block containing $A$ upon postselection on $|0\rangle^{\otimes a}$.
    """)

with col2:
    st.markdown("**QSVT Circuit Structure**")
    st.latex(r"\tilde{U} = e^{i\phi_0 Z} \prod_{j=1}^{d} \left[ U_A \cdot e^{i\phi_j Z} \right]")
    st.markdown(r"""
    The angles $\{\phi_j\}_{j=0}^d$ encode a degree-$d$ polynomial $P$ acting on the singular values of $A$.
    """)

st.markdown("---")

# Complexity Analysis
st.header("Computational Complexity")

st.info(r"**Classical vs Quantum Resource Scaling** — For an $N$-dimensional system encoded in $n = \log_2 N$ qubits:")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Classical Complexity**")
    st.markdown(r"""
    | Operation | Cost |
    |-----------|------|
    | Store state | $O(N)$ memory |
    | Dense matrix-vector | $O(N^2)$ |
    | Sparse matrix-vector | $O(N \cdot s)$, $s$ = sparsity |
    | Time evolution $e^{-iHt}$ | $O(N \cdot t / \epsilon)$ |
    """)

with col_right:
    st.markdown("**Quantum (QSVT) Complexity**")
    st.markdown(r"""
    | Operation | Cost |
    |-----------|------|
    | Encode state | $O(n)$ qubits |
    | Block-encoded query | $O(1)$ per call |
    | Polynomial $P(A)$ of degree $d$ | $O(d)$ queries to $U_A$ |
    | Total gate complexity | $O(d \cdot \text{poly}(n))$ |
    """)

st.latex(r"\text{Quantum advantage: } O(d \cdot \text{poly}(\log N)) \text{ vs } O(N)")

st.markdown("---")

# Demonstrations
st.header("Demonstrations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **1D Advection-Diffusion**
    
    Solve $\\partial_t u = \\nu \\partial_x^2 u - c \\partial_x u$
    
    - LCU block-encoding of Laplacian
    - QSVT for $e^{(A-I)t}$ evolution
    - QFT-based advection operator
    
    *Navigate via sidebar*
    """)

with col2:
    st.markdown("""
    **2D Diffusion Extension**
    
    5-point stencil Laplacian:
    
    - $n_x + n_y$ qubits for grid
    - Tensor product structure
    - 2D visualization
    
    *Navigate via sidebar*
    """)

with col3:
    st.markdown("""
    **Numerical Integration**
    
    Compute $\\int_a^b f(x)\\,dx$ via overlap:
    
    - Compute-Uncompute: $O(n)$
    - Arithmetic Comparator: $O(Mn)$  
    - QSVT Parity: $O(dn)$
    
    *Navigate via sidebar*
    """)

st.markdown("---")

# Technical Details
with st.expander("**Algorithm Details: QSVT for Time Evolution**", expanded=False):
    st.markdown(r"""
    ### Target Function and Polynomial Approximation
    
    For time evolution under operator $A$, we approximate:
    $$f(x) = e^{t(x-1)} \approx \sum_{k=0}^{d} c_k T_k(x)$$
    
    where $T_k$ are Chebyshev polynomials. The degree $d \sim O(t)$ for fixed accuracy $\epsilon$.
    
    ### Parity Constraint
    
    QSVT requires polynomials with definite parity. We use:
    $$\tilde{f}(x) = e^{t(|x|-1)}$$
    
    This is an **even function**, satisfying $\tilde{f}(x) = \tilde{f}(-x)$.
    
    ### Angle Computation
    
    Given Chebyshev coefficients $\{c_k\}$, the QSVT angles $\{\phi_j\}$ are computed via:
    1. Laurent polynomial factorization
    2. Root finding in the complex plane
    3. Prony's method or SDP optimization
    
    Complexity: $O(d^3)$ classical preprocessing.
    """)

with st.expander("**Limitations and Caveats**", expanded=False):
    st.markdown(r"""
    | Issue | Impact | Mitigation |
    |-------|--------|------------|
    | Postselection overhead | Success prob. $\sim 1/\alpha^2$ | Amplitude amplification |
    | State preparation | $O(N)$ for arbitrary states | Assume efficient structure |
    | Readout | Full state tomography is $O(N)$ | Extract only observables |
    | NISQ noise | Gate errors accumulate | Error mitigation, fault tolerance |
    | Angle precision | Numerical instability for large $d$ | Robust optimization methods |
    """)

# Footer
st.markdown("---")
st.markdown("**Implementation:** Qiskit 1.x with AerSimulator | **Method:** QSVT with LCU Block Encoding")
st.caption("QHack 2026 — Quantum Algorithms for Scientific Computing")