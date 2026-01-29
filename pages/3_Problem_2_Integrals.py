import streamlit as st
from measurements import compare_integral_methods

st.set_page_config(page_title="Problem 2: Integral Estimation", layout="wide")
st.title("Problem 2: Quantum Integral Estimation")

st.markdown(r"""
## Theory: Quantum Integral Estimation

### The Problem
Estimate $\int_a^b f(x)\,dx$ using a quantum computer with $n$ qubits representing $N = 2^n$ grid points.

**Classical Baseline:** Riemann sum
$$I_{\text{classical}} = \frac{1}{N} \sum_{j=0}^{N-1} f(x_j) \cdot \mathbb{1}_{[a,b]}(x_j)$$
where $x_j = j/N$ and $\mathbb{1}_{[a,b]}$ is the indicator function.

---

### Naive Approach: Why It Fails ❌

**Failed Strategy: Controlled StatePreparation**

1. Encode function into quantum state: $|\psi_f\rangle = \frac{1}{\|f\|} \sum_j f(x_j) |j\rangle$
2. Encode indicator into quantum state: $|\psi_g\rangle = \frac{1}{\|g\|} \sum_j \mathbb{1}_{[a,b]}(x_j) |j\rangle$
3. Build controlled-$U_g^\dagger U_f$ gate to measure overlap $\langle 0|U_g^\dagger U_f|0\rangle$

**Why it crashes:**
- `StatePreparation` gates for large $n$ require materializing $2^n \times 2^n$ unitary matrices in memory
- Controlling these gates means computing controlled versions of the full matrix
- At $n \approx 8$, this requires $2^{16}$ complex numbers = **~2 GB per matrix** → Segmentation Fault

**Key Lesson:** Never try to control large state preparation gates directly!

---

### Elegant Solution: Gate-Based Windowing ✅

**Our Strategy: Use the Initialize Gate for Arbitrary Intervals**

The key breakthrough is using Qiskit's `Initialize` gate instead of `StatePreparation`:

**1. Data Loading (Initialize)** ← Safe & Efficient
- Use Qiskit's `Initialize` gate to prepare arbitrary states
- This gate efficiently implements state prep **without materializing full matrices**
- Memory: $O(2^n)$ instead of $O(2^{2n})$ for controlled unitaries

**2. Arbitrary Window Encoding**
Instead of struggling with controlled gates, we build the indicator state classically:
$$|\psi_g\rangle = \frac{1}{\|g\|} \sum_{j: x_j \in [a,b]} |j\rangle$$

Then use `Initialize` to prepare this state as a quantum gate.

**3. Overlap Measurement via Uncomputation**
Apply the inverse of the indicator preparation:
$$U_g^\dagger = \text{Initialize}(g)^\dagger$$

This projects back to the computational basis.

**4. Measurement**
$$P(\text{all zeros}) = |\langle 0^n | U_g^\dagger U_f |0\rangle|^2$$

**3. Measurement**
$$P(\text{all zeros}) = |\langle 0^n | U_g^\dagger U_f |0\rangle|^2$$

$$I_{\text{quantum}} = \sqrt{P(\text{all zeros})} \cdot \|f\| \cdot \|g\| \cdot \frac{1}{N}$$

---

### Why This Works for ANY Interval

| Aspect | Naive Approach | Dyadic Only | Our Solution |
|--------|---|---|---|
| **Window Encoding** | Controlled StatePrep | Hadamard gates | Initialize gate |
| **Memory** | $O(2^{2n})$ | $O(n)$ | $O(2^n)$ |
| **Segfault Risk** | YES (n > 8) | NO (n ≤ 12) | NO (n ≤ 12+) |
| **Intervals** | Arbitrary [a,b] | Dyadic only | **Arbitrary [a,b]** |
| **Scalability** | Limited | Good | Excellent |

---

### Mathematical Summary for Arbitrary [a, b]

$$\int_a^b f(x)\,dx \approx \underbrace{\sqrt{P(\text{0}^n)}}_{\text{overlap}} \cdot \underbrace{\|f\|}_{\text{func norm}} \cdot \underbrace{\|g\|}_{\text{window norm}} \cdot \underbrace{\frac{1}{2^n}}_{\text{grid spacing}}$$

where $g_j = \begin{cases} 1 & \text{if } x_j \in [a,b] \\ 0 & \text{otherwise} \end{cases}$

**Key Insight:** The `Initialize` gate efficiently prepares **any** normalized quantum state, making arbitrary intervals tractable!
""")

st.markdown("---")

st.header("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    func_name = st.selectbox("Function", ["sin", "gaussian", "linear"], 
                             help="sin: sin(2πx)+2, gaussian: exp(-20(x-0.5)²), linear: 2x+0.5")
    
with col2:
    n_qubits = st.slider("Qubits (n)", min_value=3, max_value=12, value=4,
                        help="Grid resolution: 2^n points")
    
with col3:
    shots = st.slider("Shots", min_value=1000, max_value=100000, value=10000, step=1000,
                     help="Number of measurement shots")

st.subheader("Integration Interval")
st.markdown("**Select a dyadic interval (supported by Hadamard gate windowing):**")

interval_choice = st.radio("Interval", 
    ["Full: [0, 1]", "Left Half: [0, 0.5]", "Right Half: [0.5, 1]"],
    help="These dyadic intervals can be efficiently measured using only Hadamard and X gates")

# Map radio selection to (a, b) tuple
interval_map = {
    "Full: [0, 1]": (0.0, 1.0),
    "Left Half: [0, 0.5]": (0.0, 0.5),
    "Right Half: [0.5, 1]": (0.5, 1.0)
}
a, b = interval_map[interval_choice]

st.markdown(f"**Selected interval: [{a}, {b}]**")

st.markdown("---")

if st.button("Run Computation", type="primary"):
    with st.spinner("Computing integral estimate..."):
        try:
            results = compare_integral_methods(n_qubits, a, b, func_name, shots)
            
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Exact Value", f"{results['exact']:.6f}")
            
            with col2:
                st.metric("Quantum Estimate", f"{results['compute_uncompute']:.6f}",
                         delta=f"Error: {results['error']:.6f}")
            
            # Results table
            st.dataframe(
                {
                    "Method": ["Exact", "Quantum Estimate"],
                    "Value": [f"{results['exact']:.6f}", 
                             f"{results['compute_uncompute']:.6f}"],
                    "Error": ["0.000000", 
                            f"{results['error']:.6f}"],
                    "Relative Error %": ["0.00", 
                                        f"{100*results['error']/max(abs(results['exact']), 1e-10):.2f}"]
                },
                use_container_width=True
            )
            
            st.success("Computation complete!")
            
            st.info(f"""
            **Summary for $\\int_{{{a}}}^{{{b}}} f(x)\\,dx$:**
            - Grid resolution: $2^{n_qubits} = {2**n_qubits}$ points
            - Interval type: {results['type']}
            - Exact integral: {results['exact']:.6f}
            - Quantum estimate: {results['compute_uncompute']:.6f}
            - Absolute error: {results['error']:.6f}
            """)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.warning("""
            **Troubleshooting:**
            - Ensure interval is one of the supported dyadic intervals
            - Try reducing qubits if memory-constrained
            """)
