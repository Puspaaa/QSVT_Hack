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

**4. Classical preprocessing is essential**  
Computing QSVT angles ($\{\phi_j\}$) from the target polynomial is a nontrivial
classical optimization problem — the quantum algorithm alone is not enough.
""")

    with col2:
        st.markdown("### What We Built")
        st.markdown("""
| Component | Status |
|-----------|--------|
| 1D advection-diffusion solver | Done |
| 2D advection-diffusion solver | Done |
| Quantum integration (3 methods) | Done |
| LCU block encoding | Done |
| QSVT circuit construction | Done |
| Chebyshev polynomial optimization | Done |
| Angle finding | Done |
| 3-way validation (exact/classical/quantum) | Done |
| Interactive visualization | Done |
""")

        st.markdown("### Future Directions")
        st.markdown("""
- Error mitigation for NISQ hardware
- Higher-order Trotter splitting
- Amplitude estimation for integration
- Extension to nonlinear PDEs
- Fault-tolerant resource estimation
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
