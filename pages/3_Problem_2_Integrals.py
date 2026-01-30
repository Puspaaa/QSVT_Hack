import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial.chebyshev import chebval
from measurements import run_overlap_integral, run_qsvt_integral_arbitrary, run_arithmetic_integral, get_function_data, get_boxcar_targets
from solvers import robust_poly_coef, Angles_Fixed

st.set_page_config(page_title="Problem 2: Quantum Integration", layout="wide")

st.title("Problem 2(a): Quantum Numerical Integration")

# =============================================================================
# PROBLEM STATEMENT
# =============================================================================
st.markdown(r"""
**Objective:** Estimate the integral $I = \int_a^b f(x)\,dx$ using a quantum register of $n$ qubits.

**Discretization:** The grid consists of $N = 2^n$ points with spacing $\Delta x = 1/N$:
$$x_j = j\,\Delta x, \quad j = 0,1,\dots,N-1$$
""")

st.header("Quantum Formulation")

col_form1, col_form2 = st.columns(2)

with col_form1:
    st.markdown(r"""
    **Function state** (from Problem 1):
    $$|u\rangle = \frac{1}{\|u\|}\sum_{j=0}^{N-1} u(x_j, T)\,|j\rangle$$
    """)

with col_form2:
    st.markdown(r"""
    **Interval indicator state:**
    $$|D\rangle = \frac{1}{\sqrt{|D|}}\sum_{j\in D} |j\rangle$$
    where $D = \{j : a \le j\,\Delta x \le b\}$
    """)

st.info(r"""
**Key Result:** The integral reduces to an inner product:
$$\int_a^b u(x,T)\,dx = \langle D|u\rangle \cdot \frac{\sqrt{|D|}\,\|u\|}{N}$$

The problem therefore reduces to estimating the overlap $\langle D|u\rangle$, accessible via Hadamard test or amplitude estimation.
""")

st.markdown("---")

# =============================================================================
# THE THREE METHODS
# =============================================================================
st.header("Three Quantum Approaches")

method_col1, method_col2, method_col3 = st.columns(3)

with method_col1:
    st.subheader("Method 1: Compute-Uncompute")
    st.markdown("""
    **Best for:** Special intervals
    
    - Half-intervals: [0, 0.5], [0.25, 0.75]
    - O(n) gates only
    - < 1% error
    
    *Fast but limited intervals*
    """)

with method_col2:
    st.subheader("Method 2: Arithmetic/Comparison")
    st.markdown("""
    **Arbitrary intervals [a, b]**
    
    - MCX-based comparator circuit
    - Complexity: O(M·n) gates
    - Typical error: 3-15%
    
    *General but higher gate count*
    """)

with method_col3:
    st.subheader("Method 3: QSVT Boxcar")
    st.markdown("""
    **Polynomial approximation**
    
    - Even/odd parity decomposition
    - Complexity: O(d·n) gates
    - Typical error: 5-30%
    
    *Smooth filter approximation*
    """)

# =============================================================================
# EXPANDABLE THEORY SECTIONS
# =============================================================================
st.markdown("---")
st.header("Technical Details")

with st.expander("**State Encoding and Inner Product Formulation**", expanded=False):
    st.markdown(r"""
    ### Discretization
    
    With $n$ qubits, discretize $[0,1)$ into $N = 2^n$ grid points: $x_j = j/N$
    
    **Function state:**
    $$|f\rangle = \frac{1}{\|f\|} \sum_{j=0}^{N-1} f(x_j) |j\rangle, \quad \|f\| = \sqrt{\sum_j |f(x_j)|^2}$$
    
    **Indicator state:**
    $$|\chi_D\rangle = \frac{1}{\sqrt{|D|}} \sum_{j \in D} |j\rangle$$
    
    ### Integral as Inner Product
    
    The Riemann sum becomes:
    $$I = \int_D f(x)\,dx \approx \Delta x \sum_{j \in D} f(x_j) = \|f\| \cdot \sqrt{|D|} \cdot \Delta x \cdot \langle \chi_D | f \rangle$$
    
    The overlap $|\langle \chi_D | f \rangle|^2$ is measurable via amplitude estimation.
    """)

with st.expander("**Constructing the Interval State |D⟩**", expanded=False):
    st.markdown(r"""
    ### Oracle-Based Interval Marking
    
    The set of valid indices $D$ is characterized by a Boolean function:
    $$\chi_D(j) = \begin{cases} 1 & \text{if } a \le j\,\Delta x \le b \\ 0 & \text{otherwise} \end{cases}$$
    
    Since this involves only arithmetic comparisons, it can be efficiently implemented as a quantum oracle:
    $$O_D\,|j\rangle|i\rangle = |j\rangle\,|i \oplus \chi_D(j)\rangle$$
    
    ### Post-Selection from Uniform Superposition
    
    Starting from the uniform superposition $|+\rangle^{\otimes n}$, measuring the ancilla after applying $O_D$ yields outcome 1 with probability:
    $$\Pr(1) = \frac{|D|}{N}$$
    
    Conditioned on this outcome, the register collapses to the desired state $|D\rangle$.
    
    The expected number of repetitions required is:
    $$O\left(\frac{N}{|D|}\right) = O\left(\frac{1}{b-a}\right)$$
    
    ### Speedup via Grover Amplitude Amplification
    
    Apply Grover-style amplitude amplification using:
    - Oracle $O_D$ to mark indices in $D$
    - Reflection about $|+\rangle^{\otimes n}$
    
    After amplification, the state rotates close to the target subspace:
    $$|+\rangle^{\otimes n} \longrightarrow |D_{\text{approx}}\rangle = \alpha\,|D\rangle + \sqrt{1-\alpha^2}\,|g\rangle, \quad \alpha \approx 1$$
    
    where $|g\rangle$ has support only on indices outside $D$.
    
    Measuring the ancilla now yields outcome 1 with significantly higher probability, reducing complexity to:
    $$O\left(\sqrt{\frac{N}{|D|}}\right) = O\left(\frac{1}{\sqrt{b-a}}\right)$$
    
    This is a **quadratic speedup** over naive post-selection.
    """)

with st.expander("**Method 1: Compute-Uncompute Circuit**", expanded=False):
    st.markdown(r"""
    ### Applicable Intervals
    
    Works efficiently for intervals aligned with qubit structure:
    
    | Interval | Preparation | Circuit |
    |----------|-------------|---------|
    | $[0, 0.5]$ (Left Half) | H on all qubits except MSB | `H⊗(n-1) ⊗ \|0⟩` |
    | $[0.25, 0.75]$ (Middle Half) | H + CNOT pattern | Entangled state |
    
    ### Circuit
    
    $$|0\rangle^{\otimes n} \xrightarrow{U_f} |f\rangle \xrightarrow{U_D^\dagger} \xrightarrow{\text{Measure}}$$
    
    Probability of measuring all zeros: $P(0^n) = |\langle \chi_D | f \rangle|^2$
    
    ### Complexity
    - Gate count: $O(n)$ for special intervals
    - Error: < 1% (limited by shot noise)
    - Limitation: Only specific interval structures
    """)

with st.expander("**Method 2: Arithmetic Comparator Circuit**", expanded=False):
    st.markdown(r"""
    ### Oracle Implementation
    
    This method implements the interval oracle $O_D$ using arithmetic comparison circuits.
    The Boolean function $\chi_D(j)$ is realized by checking $a \le j \le b$:
    
    $$|j\rangle|0\rangle_{\text{anc}} \xrightarrow{O_D} |j\rangle|\chi_D(j)\rangle_{\text{anc}}$$
    
    ### Two Implementation Approaches
    
    **Approach A: MCX per state** (current implementation)
    - For each $j \in [a_{int}, b_{int}]$, apply MCX with control state = binary(j)
    - Simple but scales as $O(M \cdot n)$ where $M$ = interval size
    
    **Approach B: IntegerComparator** (more efficient)
    - Use Qiskit's `IntegerComparator` for $x \ge a$ and $x \le b$
    - Combine results: `target = (x >= a) AND (x <= b)`
    - Scales as $O(n)$ comparator circuits
    
    ### Circuit Flow
    
    ```
    |0⟩^n   ──[H^⊗n]──[Comparator Oracle]──[U_f†]──[Measure]
    |0⟩_anc ─────────────────────┘
    ```
    
    1. Create uniform superposition: $|+\rangle^{\otimes n} = \frac{1}{\sqrt{N}}\sum_j |j\rangle$
    2. Apply oracle to mark states in interval (flip ancilla if $j \in [a,b]$)
    3. Apply inverse state preparation $U_f^\dagger$
    4. Measure and postselect on ancilla = 1
    
    ### Post-Selection Analysis
    
    The joint probability of measuring main register = 0 and ancilla = 1:
    $$P(\text{main}=0, \text{mark}=1) = \frac{|D|}{N} \cdot |\langle D | f \rangle|^2$$
    
    The post-selection rate (probability of ancilla = 1):
    $$\Pr(\text{mark}=1) = \frac{|D|}{N}$$
    
    ### Complexity
    - **MCX approach:** $O(M \cdot n)$ gates where $M = |D|$ points
    - **IntegerComparator:** $O(n)$ gates (more efficient)
    - **Post-selection overhead:** $O(N/|D|) = O(1/(b-a))$ repetitions
    - **With amplitude amplification:** $O(\sqrt{N/|D|})$ — quadratic speedup
    - **Error:** 3-15% (depends on interval alignment and shot noise)
    """)

with st.expander("**Method 3: QSVT Boxcar Filter**", expanded=False):
    st.markdown(r"""
    ### Approach
    
    Approximate the indicator function $\chi_{[a,b]}(x)$ with polynomials:
    
    $$\chi_{[a,b]}(x) \approx P_{\text{even}}(x) + P_{\text{odd}}(x)$$
    
    where:
    - $P_{\text{even}}(x) = \frac{1}{2}[\chi(x) + \chi(-x)]$ — symmetric part
    - $P_{\text{odd}}(x) = \frac{1}{2}[\chi(x) - \chi(-x)]$ — antisymmetric part
    
    ### Parity Requirement
    
    QSVT implements polynomials of definite parity:
    - Even: $P(-x) = P(x)$
    - Odd: $P(-x) = -P(x)$
    
    The boxcar has no parity, so we decompose and run two circuits.
    
    ### Block Encoding
    
    Encode the position operator via $\cos(\pi x)$:
    $$\langle j | \cos(\pi \hat{X}) | k \rangle = \cos(\pi j / N) \delta_{jk}$$
    
    Maps $x \in [0, 1)$ to eigenvalues $\lambda \in [-1, 1]$.
    
    ### Circuit
    ```
    |f> --[QSVT_even]--[Measure]-- -> amp_even
    |f> --[QSVT_odd]--[Measure]--  -> amp_odd
    
    Total = amp_even + amp_odd
    ```
    
    ### Complexity
    - Gate count: $O(d \cdot n)$ where $d$ = polynomial degree
    - Error: 5-30% (depends on Gibbs phenomenon at discontinuities)
    - Advantage: Leverages full QSVT framework
    """)

st.markdown("---")

# =============================================================================
# INTERACTIVE DEMO SECTION
# =============================================================================
st.markdown("### Configuration")

# Use tabs for organization
config_tab, viz_tab = st.tabs(["Setup", "Function Preview"])

with config_tab:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Function to Integrate**")
        
        func_type = st.radio(
            "Function Type",
            ["Preset Functions", "Custom Function"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if func_type == "Preset Functions":
            func_choice = st.selectbox(
                "Select f(x)", 
                ["sin", "gaussian", "linear", "quadratic", "step", "sawtooth", "cosine"],
                help="Choose from preset test functions"
            )
            custom_func = None
            
            # Show function formula
            func_formulas = {
                "sin": r"$f(x) = \sin(2\pi x) + 2$",
                "gaussian": r"$f(x) = e^{-20(x-0.5)^2}$",
                "linear": r"$f(x) = 2x + 0.5$",
                "quadratic": r"$f(x) = 4x(1-x) + 0.5$",
                "step": r"$f(x) = \begin{cases} 1 & x < 0.5 \\ 2 & x \geq 0.5 \end{cases}$",
                "sawtooth": r"$f(x) = (x \mod 0.25) \times 4 + 0.5$",
                "cosine": r"$f(x) = \cos(2\pi x) + 1.5$"
            }
            st.latex(func_formulas.get(func_choice, ''))
            
        else:
            st.markdown("**Python expression (use x as variable):**")
            custom_func = st.text_input(
                "Custom f(x)", 
                value="np.sin(3*np.pi*x) + 1.5",
                help="Use numpy functions with 'np.' prefix. x in [0,1]."
            )
            func_choice = "custom"
            
            # Validate custom function
            try:
                x_test = np.linspace(0, 1, 10)
                y_test = eval(custom_func)
                if np.any(y_test < 0):
                    st.warning("Warning: Function has negative values. Results may be less accurate.")
                st.success("Function syntax is valid")
            except Exception as e:
                st.error(f"Invalid function: {e}")
                custom_func = None
        
        # Quantum parameters in a nice box
        st.markdown("**Quantum Parameters**")
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            n_qubits = st.slider("Qubits (n)", 4, 10, 6)
        with param_col2:
            shots = st.select_slider("Shots", options=[1000, 5000, 10000, 20000, 50000], value=10000)
        
        st.caption(f"Grid: {2**n_qubits} points, dx = {1/2**n_qubits:.6f}")

    with col2:
        st.markdown("**Method and Interval**")
        
        method = st.radio(
            "Integration Method", 
            ["Compute-Uncompute (Special Intervals)", 
             "Arithmetic/Comparison (Any Interval)",
             "QSVT Parity Decomposition"],
            help="Choose the quantum algorithm"
        )
        
        # Method-specific interval selection
        if "Compute-Uncompute" in method:
            method = "Compute-Uncompute (Special Intervals)"
            st.caption("*Efficient for half-intervals*")
            interval_choice = st.radio(
                "Select Domain D",
                ["Left Half [0, 0.5]", "Middle Half [0.25, 0.75]"],
                help="These intervals have efficient O(n) state preparation"
            )
            if "Left" in interval_choice:
                interval_id = "left_half"
                a_val, b_val = 0.0, 0.5
            else:
                interval_id = "middle_half"
                a_val, b_val = 0.25, 0.75
                
        elif "Arithmetic" in method:
            method = "Arithmetic/Comparison (Arbitrary Intervals)"
            st.caption("*Uses efficient IntegerComparator oracle*")
            a_val, b_val = st.slider(
                "Select Interval [a, b]", 0.0, 1.0, (0.25, 0.75), 
                step=0.01, key="arith_slider"
            )
            interval_id = None
            
            # Grover amplitude amplification option
            st.markdown("**Advanced Options**")
            use_grover = st.checkbox(
                "Enable Grover Amplitude Amplification",
                value=False,
                help="Quadratic speedup for post-selection: O(√(N/M)) vs O(N/M)"
            )
            if use_grover:
                grover_iterations = st.slider(
                    "Grover Iterations (0 = auto)",
                    0, 5, 0,
                    help="0 = automatically calculate optimal iterations"
                )
            else:
                grover_iterations = 0
            
        else:  # QSVT
            method = "QSVT Parity Decomposition (Arbitrary)"
            st.caption("*Polynomial approximation*")
            a_val, b_val = st.slider(
                "Select Interval [a, b]", 0.0, 1.0, (0.3, 0.7),
                step=0.01
            )
            interval_id = None
            use_grover = False
            grover_iterations = 0

# Initialize Grover variables for methods that don't use them
if "Arithmetic" not in method:
    use_grover = False
    grover_iterations = 0

# Helper function for plotting (defined outside tabs)
def get_function_for_plot(func_name, n, custom_expr=None):
    """Get function values for plotting."""
    N = 2**n
    x = np.linspace(0, 1, N, endpoint=False)
    
    if func_name == "custom" and custom_expr is not None:
        try:
            y = eval(custom_expr)
            if np.isscalar(y):
                y = np.full_like(x, y)
        except:
            y = np.ones_like(x)
    elif func_name == "sin":
        y = np.sin(2*np.pi*x) + 2.0
    elif func_name == "gaussian":
        y = np.exp(-20*(x-0.5)**2)
    elif func_name == "linear":
        y = 2*x + 0.5
    elif func_name == "quadratic":
        y = 4*x*(1-x) + 0.5
    elif func_name == "step":
        y = np.where(x < 0.5, 1.0, 2.0)
    elif func_name == "sawtooth":
        y = (x % 0.25) * 4 + 0.5
    elif func_name == "cosine":
        y = np.cos(2*np.pi*x) + 1.5
    else:
        y = np.ones_like(x)
    
    return x, y

with viz_tab:
    st.markdown("#### Function Preview with Integration Domain")
    
    # Create visualization
    x_plot, y_plot = get_function_for_plot(func_choice, n_qubits, custom_func)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot full function
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')

    # Highlight integration domain
    mask = (x_plot >= a_val) & (x_plot < b_val)
    ax.fill_between(x_plot, 0, y_plot, where=mask, alpha=0.3, color='green', 
                    label=f'∫ Domain [{a_val:.2f}, {b_val:.2f}]')

    # Mark interval boundaries
    ax.axvline(x=a_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=b_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Compute exact integral for display
    dx = 1.0 / len(x_plot)
    exact_integral = np.sum(y_plot[mask]) * dx

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'f(x) over [{a_val:.2f}, {b_val:.2f}] — Exact ∫ ≈ {exact_integral:.4f}', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(y_plot) * 1.1)

    st.pyplot(fig)
    plt.close()
    
    # Grid info
    N = 2**n_qubits
    a_int = int(np.floor(a_val * N))
    b_int = min(int(np.floor(b_val * N)), N-1)
    st.caption(f"Grid: {N} points | Interval indices [{a_int}, {b_int}] ({b_int - a_int + 1} points)")

# =============================================================================
# EXECUTION
# =============================================================================
st.markdown("---")

# Run button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_button = st.button("Run Quantum Integration", type="primary", use_container_width=True)

if run_button:
    # Validate inputs
    if func_choice == "custom" and custom_func is None:
        st.error("Please enter a valid custom function!")
    else:
        with st.spinner("Running quantum simulation..."):
            
            # Use preset function name or "sin" as base for custom
            effective_func = func_choice if func_choice != "custom" else "sin"
            
            if method == "Compute-Uncompute (Special Intervals)":
                res = run_overlap_integral(n_qubits, effective_func, interval_id, shots)
                
                # If custom function, recalculate the exact integral
                if func_choice == "custom":
                    x_grid, y_vals = get_function_for_plot("custom", n_qubits, custom_func)
                    N = 2**n_qubits
                    dx = 1/N
                    if interval_id == "left_half":
                        mask = x_grid < 0.5
                    else:
                        mask = (x_grid >= 0.25) & (x_grid < 0.75)
                    res['integral_exact'] = np.sum(y_vals[mask]) * dx
                
                st.success("Compute-Uncompute Complete")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Exact Integral", f"{res['integral_exact']:.5f}")
                col2.metric("Quantum Estimate", f"{res['integral_est']:.5f}")
                rel_err = abs(res['error']/res['integral_exact'])*100 if res['integral_exact'] != 0 else 0
                col3.metric("Relative Error", f"{rel_err:.2f}%")
                
                st.markdown(f"""
                **Results Analysis**
                
                | Metric | Value |
                |--------|-------|
                | Circuit Depth | {res['depth_window']} |
                | Gate Count | {res['gate_count_window']} |
                | Overlap $|\\langle\\chi_D|f\\rangle|^2$ | {(res['integral_est']/(res['integral_exact']+1e-10))**2:.4f} |
                
                The interval state can be prepared with O(n) = O({n_qubits}) gates due to qubit structure alignment.
                """)
            
            elif method == "Arithmetic/Comparison (Arbitrary Intervals)":
                res = run_arithmetic_integral(n_qubits, effective_func, a_val, b_val, shots,
                                              use_amplitude_amplification=use_grover,
                                              num_grover_iterations=grover_iterations)
                
                if func_choice == "custom":
                    x_grid, y_vals = get_function_for_plot("custom", n_qubits, custom_func)
                    N = 2**n_qubits
                    dx = 1/N
                    mask = (x_grid >= a_val) & (x_grid < b_val)
                    res['integral_exact'] = np.sum(y_vals[mask]) * dx
                
                if res.get('use_grover', False):
                    st.success(f"Arithmetic/Comparison with Grover Amplification ({res['grover_iterations']} iterations)")
                else:
                    st.success("Arithmetic/Comparison Method Complete (IntegerComparator Oracle)")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Exact [{a_val:.2f}, {b_val:.2f}]", f"{res['integral_exact']:.5f}")
                col2.metric("Quantum Estimate", f"{res['integral_est']:.5f}")
                rel_err = abs(res['error']/res['integral_exact'])*100 if res['integral_exact'] != 0 else 0
                col3.metric("Relative Error", f"{rel_err:.1f}%")
                
                # Different output based on whether Grover was used
                if res.get('use_grover', False):
                    st.markdown(f"""
                    **Algorithm with Grover Amplitude Amplification**
                    
                    | Step | Operation | Result |
                    |------|-----------|--------|
                    | 1 | Apply $H^{{\\otimes {n_qubits}}}$ | Uniform superposition over {2**n_qubits} states |
                    | 2 | Oracle + Diffusion × {res['grover_iterations']} | Amplify marked states |
                    | 3 | Apply $U_f^\\dagger$ | Compute overlap |
                    | 4 | Measure | Extract amplified result |
                    
                    **Grover Amplification Details**
                    
                    | Parameter | Value |
                    |-----------|-------|
                    | Grover Iterations | {res['grover_iterations']} |
                    | Expected Speedup | $O(\\sqrt{{N/M}}) = O(\\sqrt{{{2**n_qubits}/{res['num_points']}}}) \\approx {np.sqrt(2**n_qubits/max(1,res['num_points'])):.1f}\\times$ |
                    | P(marked) after amplification | {res['post_select_rate']:.2%} |
                    """)
                else:
                    st.markdown(f"""
                    **Algorithm Execution Details**
                    
                    | Step | Operation | Result |
                    |------|-----------|--------|
                    | 1 | Apply $H^{{\\otimes {n_qubits}}}$ | Uniform superposition over {2**n_qubits} states |
                    | 2 | IntegerComparator Oracle | Mark {res['num_points']} states in {res['interval_int']} |
                    | 3 | Apply $U_f^\\dagger$ | Compute overlap with $|f\\rangle$ |
                    | 4 | Measure & post-select | Keep target=1 results |
                    """)
                
                st.markdown(f"""
                **Measurement Statistics**
                
                | Quantity | Value | Meaning |
                |----------|-------|---------|
                | P(target=1) | {res['post_select_rate']:.2%} | Post-selection rate |
                | P(main=0, target=1) | {res['p_zero_and_marked']:.4f} | Joint probability |
                | $|\\langle\\chi_D|f\\rangle|$ | {res['overlap']:.4f} | Extracted overlap |
                
                **Circuit:** Depth = {res['depth']}, Gates = {res['gate_count']}, Oracle = {res.get('oracle_type', 'IntegerComparator')}
                """)
                
                # =============================================================
                # STATE MARKING VISUALIZATION (inspired by Miro's notebook)
                # =============================================================
                st.markdown("---")
                st.markdown("#### State Marking Visualization")
                
                with st.expander("**View Marked States and Amplitudes**", expanded=True):
                    # Compute state amplitudes for visualization
                    N_vis = 2**n_qubits
                    x_grid_vis = np.linspace(0, 1, N_vis, endpoint=False)
                    
                    # Get function amplitudes
                    if func_choice == "custom" and custom_func is not None:
                        _, y_vals_vis = get_function_for_plot("custom", n_qubits, custom_func)
                    else:
                        _, y_vals_vis = get_function_for_plot(effective_func, n_qubits, None)
                    
                    # Normalize
                    norm_val = np.linalg.norm(y_vals_vis)
                    if norm_val > 0:
                        amplitudes = y_vals_vis / norm_val
                    else:
                        amplitudes = y_vals_vis
                    
                    # Determine which states are in range
                    a_int = int(np.floor(a_val * N_vis))
                    b_int = min(int(np.floor(b_val * N_vis)), N_vis - 1)
                    
                    # Create visualization tabs
                    viz_tab1, viz_tab2 = st.tabs(["Amplitude Bar Chart", "State Table"])
                    
                    with viz_tab1:
                        st.markdown("**Quantum State Amplitudes with Marked Interval**")
                        
                        fig_amp, ax_amp = plt.subplots(figsize=(12, 4))
                        
                        # Color bars based on whether they're in the marked range
                        colors = ['#2ecc71' if (a_int <= j <= b_int) else '#bdc3c7' for j in range(N_vis)]
                        
                        bars = ax_amp.bar(range(N_vis), amplitudes, color=colors, edgecolor='black', alpha=0.8)
                        
                        # Add interval boundaries
                        ax_amp.axvline(x=a_int - 0.5, color='red', linestyle='--', linewidth=2, label=f'a = {a_val:.2f} → j={a_int}')
                        ax_amp.axvline(x=b_int + 0.5, color='blue', linestyle='--', linewidth=2, label=f'b = {b_val:.2f} → j={b_int}')
                        
                        ax_amp.set_xlabel('Basis State Index |j⟩', fontsize=11)
                        ax_amp.set_ylabel('Amplitude', fontsize=11)
                        ax_amp.set_title(f'States in [{a_int}, {b_int}] are marked (green) by the comparator oracle', fontsize=11)
                        ax_amp.legend(loc='upper right')
                        ax_amp.grid(True, alpha=0.3, axis='y')
                        
                        if N_vis <= 32:
                            ax_amp.set_xticks(range(N_vis))
                        
                        st.pyplot(fig_amp)
                        plt.close()
                        
                        # Summary metrics
                        marked_sum = np.sum(amplitudes[a_int:b_int+1])
                        total_sum = np.sum(amplitudes)
                        st.markdown(f"""
                        **Marking Summary:**
                        - **Marked states:** {b_int - a_int + 1} out of {N_vis} ({100*(b_int - a_int + 1)/N_vis:.1f}%)
                        - **Sum of marked amplitudes:** {marked_sum:.4f}
                        - **Fraction of total amplitude:** {marked_sum/total_sum:.4f}
                        """)
                    
                    with viz_tab2:
                        st.markdown("**Detailed State Information**")
                        
                        # Create table similar to Miro's output
                        data_rows = []
                        for j in range(N_vis):
                            in_range = "Yes" if (a_int <= j <= b_int) else "No"
                            bin_str = format(j, f'0{n_qubits}b')
                            prob = amplitudes[j]**2
                            data_rows.append({
                                "Index |j⟩": j,
                                "Binary": bin_str,
                                "Position x": f"{x_grid_vis[j]:.4f}",
                                "Amplitude": f"{amplitudes[j]:.4f}",
                                "Probability": f"{prob:.4f}",
                                "In [a,b]": in_range
                            })
                        
                        st.dataframe(data_rows, use_container_width=True, height=300)
                
                st.markdown(f"""
                **Algorithm Execution Details**
                
                | Step | Operation | Result |
                |------|-----------|--------|
                | 1 | Apply $H^{{\\otimes {n_qubits}}}$ | Uniform superposition over {2**n_qubits} states |
                | 2 | Comparator marking | Mark {res['num_points']} states in {res['interval_int']} |
                | 3 | Apply $U_f^\\dagger$ | Compute overlap with $|f\\rangle$ |
                | 4 | Measure & post-select | Keep ancilla=1 results |
                
                **Measurement Statistics**
                
                | Quantity | Value | Meaning |
                |----------|-------|---------|
                | P(mark=1) | {res['post_select_rate']:.2%} | Post-selection rate |
                | P(main=0, mark=1) | {res['p_zero_and_marked']:.4f} | Joint probability |
                | $|\\langle\\chi_D|f\\rangle|$ | {res['overlap']:.4f} | Extracted overlap |
                
                **Circuit Complexity:** Depth = {res['depth']}, Gates = {res['gate_count']}, MCX = {res['num_points']}
                """)
                
            else:  # QSVT
                res = run_qsvt_integral_arbitrary(n_qubits, effective_func, a_val, b_val, shots)
                
                if func_choice == "custom":
                    x_grid, y_vals = get_function_for_plot("custom", n_qubits, custom_func)
                    N = 2**n_qubits
                    dx = 1/N
                    mask = (x_grid >= a_val) & (x_grid < b_val)
                    res['integral_exact'] = np.sum(y_vals[mask]) * dx
                
                st.success("QSVT Parity Decomposition Complete")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Exact [{a_val:.2f}, {b_val:.2f}]", f"{res['integral_exact']:.5f}")
                col2.metric("QSVT Estimate", f"{res['integral_est']:.5f}")
                col3.metric("Poly Degrees", f"Even:{res['deg_even']} / Odd:{res['deg_odd']}")
                
                st.markdown(f"""
                **Parity Decomposition Results**
                
                Boxcar indicator split into parity components:
                
                | Component | Value | Degree |
                |-----------|-------|--------|
                | Even $P_{{even}}(\\lambda)$ | {res['val_even']:.5f} | {res['deg_even']} |
                | Odd $P_{{odd}}(\\lambda)$ | {res['val_odd']:.5f} | {res['deg_odd']} |
                | **Total** | **{res['integral_est']:.5f}** | |
                
                Note: For symmetric intervals around 0.5, the odd component vanishes.
                """)
                
                # =============================================================
                # QSVT BOXCAR VISUALIZATION
                # =============================================================
                st.markdown("---")
                st.markdown("#### Boxcar Polynomial Approximation")
                
                with st.expander("**View Polynomial Approximation Details**", expanded=True):
                    # Compute visualization data
                    N_vis = 2**n_qubits
                    x_grid_vis = np.linspace(0, 1, N_vis, endpoint=False)
                    x_fine = np.linspace(0, 1, 500)
                    lambda_fine = np.linspace(-1, 1, 500)
                    lambda_grid = np.cos(np.pi * x_grid_vis)
                    lambda_from_x = np.cos(np.pi * x_fine)
                    
                    # Get target boxcar
                    t_even, t_odd, scale = get_boxcar_targets(a_val, b_val)
                    
                    # Polynomial coefficients
                    deg_even_vis = res['deg_even']
                    deg_odd_vis = res['deg_odd']
                    coef_even = robust_poly_coef(t_even, [-1, 1], deg_even_vis, epsil=2e-3)
                    coef_odd = robust_poly_coef(t_odd, [-1, 1], deg_odd_vis, epsil=2e-3)
                    
                    # Eigenvalue interval
                    l_a, l_b = np.cos(np.pi*a_val), np.cos(np.pi*b_val)
                    l_min, l_max = min(l_a, l_b), max(l_a, l_b)
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Position Space", "Eigenvalue Space", "Parity Components"])
                    
                    with tab1:
                        st.markdown("**Boxcar Window in Position Space**")
                        
                        fig1, ax1 = plt.subplots(figsize=(10, 4))
                        
                        # Ideal boxcar
                        ideal = np.where((x_fine >= a_val) & (x_fine <= b_val), scale, 0)
                        ax1.fill_between(x_fine, ideal, alpha=0.3, color='green', label='Target Interval')
                        
                        # Polynomial approximation in x-space
                        if coef_even is not None and coef_odd is not None:
                            poly_in_x = chebval(lambda_from_x, coef_even) + chebval(lambda_from_x, coef_odd)
                            ax1.plot(x_fine, poly_in_x, 'k-', linewidth=2, label='QSVT Polynomial')
                        
                        # Interval boundaries
                        ax1.axvline(a_val, color='red', linestyle='--', linewidth=2, label=f'a = {a_val}')
                        ax1.axvline(b_val, color='blue', linestyle='--', linewidth=2, label=f'b = {b_val}')
                        
                        # Grid points
                        ax1.scatter(x_grid_vis, np.zeros_like(x_grid_vis), color='purple', s=50, marker='|', label=f'Grid (N={N_vis})')
                        
                        ax1.set_xlabel('Position x ∈ [0, 1)', fontsize=12)
                        ax1.set_ylabel('Window Function', fontsize=12)
                        ax1.legend(loc='upper right')
                        ax1.set_xlim(-0.05, 1.05)
                        ax1.set_ylim(-0.2, 1.2)
                        ax1.grid(True, alpha=0.3)
                        ax1.set_title('QSVT Boxcar Approximation in Position Space')
                        
                        st.pyplot(fig1)
                        plt.close()
                    
                    with tab2:
                        st.markdown("**Eigenvalue Mapping: $\\lambda = \\cos(\\pi x)$**")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            fig2a, ax2a = plt.subplots(figsize=(5, 3.5))
                            ax2a.plot(x_fine, lambda_from_x, 'b-', linewidth=2)
                            ax2a.axhline(l_a, color='red', linestyle='--', label=f'λ(a) = {l_a:.3f}')
                            ax2a.axhline(l_b, color='blue', linestyle='--', label=f'λ(b) = {l_b:.3f}')
                            ax2a.axvline(a_val, color='red', linestyle=':', alpha=0.5)
                            ax2a.axvline(b_val, color='blue', linestyle=':', alpha=0.5)
                            ax2a.scatter(x_grid_vis, lambda_grid, color='purple', s=30, zorder=5)
                            ax2a.set_xlabel('x')
                            ax2a.set_ylabel('λ = cos(πx)')
                            ax2a.legend(fontsize=8)
                            ax2a.grid(True, alpha=0.3)
                            ax2a.set_title('x → λ Mapping')
                            st.pyplot(fig2a)
                            plt.close()
                        
                        with col_b:
                            fig2b, ax2b = plt.subplots(figsize=(5, 3.5))
                            mask_vis = (lambda_fine >= l_min) & (lambda_fine <= l_max)
                            ax2b.fill_between(lambda_fine, 0, scale, where=mask_vis, alpha=0.3, color='green')
                            
                            if coef_even is not None and coef_odd is not None:
                                poly_even = chebval(lambda_fine, coef_even)
                                poly_odd = chebval(lambda_fine, coef_odd)
                                ax2b.plot(lambda_fine, poly_even + poly_odd, 'k--', linewidth=2, label='Polynomial')
                            
                            ax2b.axvline(l_min, color='red', linestyle='--')
                            ax2b.axvline(l_max, color='blue', linestyle='--')
                            ax2b.set_xlabel('λ ∈ [-1, 1]')
                            ax2b.set_ylabel('P(λ)')
                            ax2b.legend(fontsize=8)
                            ax2b.set_xlim(-1.1, 1.1)
                            ax2b.grid(True, alpha=0.3)
                            ax2b.set_title('Boxcar in λ-space')
                            st.pyplot(fig2b)
                            plt.close()
                    
                    with tab3:
                        st.markdown("**Even/Odd Parity Decomposition**")
                        st.markdown(r"""
                        QSVT requires separate circuits for even and odd parts:
                        - **Even**: $P_{even}(\lambda) = P_{even}(-\lambda)$
                        - **Odd**: $P_{odd}(\lambda) = -P_{odd}(-\lambda)$
                        """)
                        
                        col_c, col_d = st.columns(2)
                        
                        if coef_even is not None and coef_odd is not None:
                            boxcar_even = np.array([t_even(l) for l in lambda_fine])
                            boxcar_odd = np.array([t_odd(l) for l in lambda_fine])
                            poly_even = chebval(lambda_fine, coef_even)
                            poly_odd = chebval(lambda_fine, coef_odd)
                            
                            with col_c:
                                fig3a, ax3a = plt.subplots(figsize=(5, 3.5))
                                ax3a.plot(lambda_fine, boxcar_even, 'b-', linewidth=2, alpha=0.7, label='Target')
                                ax3a.plot(lambda_fine, poly_even, 'r--', linewidth=2, label=f'Poly (d={deg_even_vis})')
                                ax3a.axvline(l_min, color='gray', linestyle=':', alpha=0.5)
                                ax3a.axvline(l_max, color='gray', linestyle=':', alpha=0.5)
                                ax3a.set_xlabel('λ')
                                ax3a.set_ylabel('P_even(λ)')
                                ax3a.set_title('Even Component')
                                ax3a.legend(fontsize=8)
                                ax3a.set_xlim(-1.1, 1.1)
                                ax3a.grid(True, alpha=0.3)
                                st.pyplot(fig3a)
                                plt.close()
                            
                            with col_d:
                                fig3b, ax3b = plt.subplots(figsize=(5, 3.5))
                                ax3b.plot(lambda_fine, boxcar_odd, 'b-', linewidth=2, alpha=0.7, label='Target')
                                ax3b.plot(lambda_fine, poly_odd, 'r--', linewidth=2, label=f'Poly (d={deg_odd_vis})')
                                ax3b.axvline(l_min, color='gray', linestyle=':', alpha=0.5)
                                ax3b.axvline(l_max, color='gray', linestyle=':', alpha=0.5)
                                ax3b.axhline(0, color='black', linewidth=0.5)
                                ax3b.set_xlabel('λ')
                                ax3b.set_ylabel('P_odd(λ)')
                                ax3b.set_title('Odd Component')
                                ax3b.legend(fontsize=8)
                                ax3b.set_xlim(-1.1, 1.1)
                                ax3b.grid(True, alpha=0.3)
                                st.pyplot(fig3b)
                                plt.close()
                
                # Discrete grid amplitudes
                with st.expander("**Discrete Grid Amplitudes**", expanded=False):
                    if coef_even is not None and coef_odd is not None:
                        poly_at_grid = chebval(lambda_grid, coef_even) + chebval(lambda_grid, coef_odd)
                        
                        # Bar chart
                        fig5, ax5 = plt.subplots(figsize=(10, 3))
                        colors = ['green' if (x >= a_val and x <= b_val) else 'lightgray' for x in x_grid_vis]
                        ax5.bar(range(N_vis), poly_at_grid, color=colors, edgecolor='black', alpha=0.8)
                        ax5.axhline(scale, color='red', linestyle='--', label=f'Target ({scale:.2f})')
                        ax5.axhline(0, color='black', linewidth=0.5)
                        ax5.set_xlabel('Grid Index |j⟩', fontsize=10)
                        ax5.set_ylabel('Amplitude P(λⱼ)', fontsize=10)
                        ax5.set_xticks(range(N_vis))
                        ax5.legend()
                        ax5.grid(True, alpha=0.3, axis='y')
                        ax5.set_title('QSVT Output Amplitudes (Green = Inside Interval)')
                        st.pyplot(fig5)
                        plt.close()
                        
                        # Table
                        data = []
                        for j in range(N_vis):
                            in_interval = "Yes" if (x_grid_vis[j] >= a_val and x_grid_vis[j] <= b_val) else "No"
                            data.append({
                                "Index |j⟩": j,
                                "Position x": f"{x_grid_vis[j]:.4f}",
                                "λ = cos(πx)": f"{lambda_grid[j]:.4f}",
                                "P(λ)": f"{poly_at_grid[j]:.4f}",
                                "In [a,b]": in_interval
                            })
                        st.dataframe(data, use_container_width=True)

# =============================================================================
# COMPARISON & Q&A (at the bottom for reference)
# =============================================================================
st.markdown("---")
st.header("Reference Materials")

ref_col1, ref_col2 = st.columns(2)

with ref_col1:
    with st.expander("**Method Comparison**", expanded=False):
        comparison_data = {
            "Method": ["Compute-Uncompute", "Arithmetic/Comparison", "QSVT Parity"],
            "Intervals": ["Special only", "Any [a,b]", "Any [a,b]"],
            "Gates": ["O(n)", "O(M·n)", "O(d·n)"],
            "Error": ["< 1%", "3-15%", "5-30%"],
            "Best For": ["Half-intervals", "Arbitrary", "Smooth approx."]
        }
        st.table(pd.DataFrame(comparison_data))

with ref_col2:
    with st.expander("**Q&A Reference**", expanded=False):
        st.markdown(r"""
        **Q: Why quantum for integration?**
        > Encode $N$ values in $\log N$ qubits. Inner products in $O(1/\varepsilon^2)$ shots.
        
        **Q: What's the speedup?**
        > Potential exponential in state prep, polynomial in measurements.
        
        **Q: Limitations?**
        > Readout is expensive. Post-selection reduces samples. NISQ noise.
        
        **Q: QSVT even/odd split?**
        > QSVT needs definite parity. Boxcar has mixed parity → split.
        """)

# Footer
st.markdown("---")
st.caption("**Note**: Method 2 (Arithmetic/Comparison) provides the clearest demonstration of arbitrary interval integration.")