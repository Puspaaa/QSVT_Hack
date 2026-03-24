"""Slide 9: Summary & References."""

import streamlit as st
from slides.components import slide_header, key_concept, reference_list, REFERENCES

TITLE = "Summary & References"


def render():
    slide_header("Summary & References", "")

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(r"""
### Key Takeaways

**1. QSVT is a unifying framework**  
Most quantum speedups — Hamiltonian simulation, matrix inversion, search, phase estimation —
are instances of applying a polynomial to singular values of a block-encoded matrix.

**2. Scientific computing is a natural application**  
PDEs and numerical integration map directly to the QSVT framework:
- Diffusion = matrix exponential via Chebyshev polynomial
- Advection = diagonal in Fourier space (QFT)
- Integration = inner product with polynomial domain indicator

**3. The pipeline is complete but the advantage is asymptotic**
Current quantum hardware is too noisy for practical advantage. Our simulator
demonstrates correctness of the algorithm. The exponential speedup activates
when $N = 2^n$ is large and we need observables, not full state readout.

*Concrete example:* for our 32-point grid, classical FD costs ~96 flops/step
and stores a 32-vector.  The quantum circuit uses 8 qubits but needs ~100K
shots per snapshot.  The crossover where quantum wins requires
$N \gg 10^6$ grid points with observable-only output (e.g. computing
$\int f(x)\,dx$ via amplitude estimation, not reading the full state).

**4. Classical preprocessing is essential**  
Computing QSVT angles ($\{\phi_j\}$) from the target polynomial is a nontrivial
classical optimization problem — the quantum algorithm alone is not enough.
""")

    st.markdown(r"""
### When Quantum Helps (Checklist)

- Large problem size ($N=2^n$) where classical memory/matvec become dominant
- Structured state preparation (oracles/circuits), not arbitrary $O(N)$ loading
- Observable-focused outputs (integrals, moments), not full tomography
- Controlled postselection/approximation error so constants do not swamp asymptotics
""")

    with col2:
        st.markdown("### What We Built")
        st.markdown("""
| Component | Status |
|-----------|--------|
| 1D advection-diffusion solver | Implemented |
| 2D advection-diffusion solver | Implemented (experimental) |
| Quantum integration (3 methods) | Implemented (QSVT parity has caveats) |
| LCU block encoding | Implemented |
| QSVT circuit construction | Implemented |
| Chebyshev polynomial optimization | Implemented |
| Angle finding | Implemented |
| 3-way validation (exact/classical/quantum) | Implemented |
| Interactive visualization | Implemented |
""")

        st.caption(
            "Known caveats are surfaced in demo slides (postselection overhead, parity-sensitive intervals, and approximation error sources)."
        )

        st.markdown("### Future Directions")
        st.markdown("""
- **Higher-order finite-difference block encodings** (see Helle et al. 2025): encoding $H = i\\beta D_{2p}$ eliminates Lie-Trotter error and improves accuracy per qubit
- Error mitigation for NISQ hardware
- Amplitude estimation for integration (replacing postselection)
- Extension to nonlinear PDEs (Carleman linearization)
- Fault-tolerant resource estimation
""")

    st.markdown("---")

    st.markdown(r"""
### Paper-Informed Context (arXiv:2512.22163)

- Recent results show that **higher-order finite differences** can outperform low-order methods
    at lower qubit/gate budgets when the problem is sufficiently smooth and well-resolved.
- Our current app remains a pedagogical baseline with explicit caveats (splitting error,
    postselection overhead, parity-sensitive cases), and now frames these tradeoffs transparently.
- Near-term project plan is phased:
    **(1)** context updates, **(2)** paper-comparison slide updates, **(3)** optional code-level migration.
""")

    st.markdown("---")

    # ── Full reference list ──
    st.markdown("### References")
    reference_list(list(REFERENCES.keys()))

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:1.5rem; margin-top:1rem;'>"
        "Thank you! Questions?"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#888;'>"
        "QHack 2026 — Quantum Algorithms for Scientific Computing"
        "</p>",
        unsafe_allow_html=True,
    )
