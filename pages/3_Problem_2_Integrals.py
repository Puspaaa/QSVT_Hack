import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial.chebyshev import chebval
from measurements import run_overlap_integral, run_qsvt_integral_arbitrary, run_arithmetic_integral, get_function_data, get_boxcar_targets
from solvers import robust_poly_coef, Angles_Fixed

st.set_page_config(page_title="Problem 2: Quantum Integration", layout="wide")
st.title("🔬 Problem 2(a): Quantum Numerical Integration")

# =============================================================================
# THEORY SECTION - Expandable
# =============================================================================
with st.expander("📚 **THEORY: Quantum Integration Overview** (Click to expand)", expanded=False):
    st.markdown(r"""
    ## The Problem: Numerical Integration
    
    We want to estimate the integral of a function $f(x)$ over a domain $D \subseteq [0,1)$:
    
    $$I = \int_D f(x)\, dx \approx \sum_{j \in D} f(x_j) \cdot \Delta x$$
    
    ### Quantum State Encoding
    
    With $n$ qubits, we discretize $[0,1)$ into $N = 2^n$ points: $x_j = j/N$ for $j = 0, 1, \ldots, N-1$.
    
    **Function state**: Encode $f(x)$ as amplitudes:
    $$|f\rangle = \frac{1}{\|f\|} \sum_{j=0}^{N-1} f(x_j) |j\rangle$$
    
    **Indicator state**: Uniform superposition over interval $D$:
    $$|\chi_D\rangle = \frac{1}{\sqrt{|D|}} \sum_{j \in D} |j\rangle$$
    
    ### The Quantum Approach
    
    The integral becomes an **inner product**:
    $$I = \langle \chi_D | f \rangle \cdot \|f\| \cdot \sqrt{|D|} \cdot \Delta x$$
    
    We measure $|\langle \chi_D | f \rangle|^2$ using quantum circuits, then extract $I$.
    """)

# Method-specific theory sections
with st.expander("🔧 **METHOD 1: Compute-Uncompute** (Special Intervals)", expanded=False):
    st.markdown(r"""
    ## Compute-Uncompute Method
    
    **Best for**: Half-intervals like $[0, 0.5]$ or $[0.25, 0.75]$ that align with qubit structure.
    
    ### Key Insight
    For certain intervals, $|\chi_D\rangle$ can be prepared with just **O(n) Hadamard gates**:
    
    | Interval | Preparation | Circuit |
    |----------|-------------|---------|
    | $[0, 0.5]$ (Left Half) | H on all qubits except MSB | `H⊗(n-1) ⊗ \|0⟩` |
    | $[0.25, 0.75]$ (Middle Half) | H + CNOT pattern | Uses entanglement |
    
    ### Circuit Structure
    ```
    |0⟩ ──[U_f]──[U_D†]──[Measure]──
    ```
    
    1. **Prepare** $|f\rangle$ from $|0\rangle^{\otimes n}$
    2. **Apply** $U_D^\dagger$ (inverse of interval state prep)
    3. **Measure** probability of $|0\rangle^{\otimes n}$
    
    $$P(0^n) = |\langle \chi_D | f \rangle|^2$$
    
    ### Advantages
    - ✅ Very efficient: O(n) gates for special intervals
    - ✅ High accuracy (< 1% error)
    - ❌ Only works for specific interval structures
    """)

with st.expander("⚙️ **METHOD 2: Arithmetic/Comparison** (Arbitrary Intervals)", expanded=False):
    st.markdown(r"""
    ## Arithmetic/Comparison Method
    
    **Best for**: Any arbitrary interval $[a, b] \subseteq [0, 1)$
    
    ### Key Insight
    Use **quantum comparators** to mark basis states within the interval:
    
    $$\text{Mark}(|j\rangle|0\rangle_{\text{anc}}) = |j\rangle|1\rangle_{\text{anc}} \quad \text{if } a \leq j < b$$
    
    ### Circuit Structure
    ```
    |0⟩^n ──[H^⊗n]──[Comparator]──[U_f†]──[Measure]──
    |0⟩_anc ─────────────┘
    ```
    
    ### Algorithm Steps
    
    1. **Uniform Superposition**: Apply $H^{\otimes n}$ to create
       $$|+\rangle = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} |j\rangle |0\rangle_{\text{anc}}$$
    
    2. **Comparator Circuit**: For each $j \in [a_{int}, b_{int}]$, apply MCX:
       $$|j\rangle|0\rangle \xrightarrow{\text{MCX}} |j\rangle|1\rangle$$
       Uses multi-controlled X gates with control state matching $j$.
    
    3. **Inverse State Prep**: Apply $U_f^\dagger$ to compute overlap
    
    4. **Measure & Post-Select**: Keep results where ancilla = 1
    
    ### Probability Analysis
    
    $$P(\text{main}=0, \text{mark}=1) = \frac{M}{N} \cdot |\langle \chi_D | f \rangle|^2$$
    
    where $M = |D|$ is the number of points in the interval.
    
    ### Advantages
    - ✅ Works for **any** interval [a, b]
    - ✅ Straightforward implementation
    - ❌ Gate count scales with interval size: O(M · n) MCX gates
    - ❌ Post-selection reduces effective sample count
    """)

with st.expander("🌀 **METHOD 3: QSVT Parity Decomposition** (Advanced)", expanded=False):
    st.markdown(r"""
    ## QSVT Parity Decomposition
    
    **Best for**: Smooth approximations of arbitrary indicator functions.
    
    ### Key Insight
    The indicator function (boxcar) $\chi_{[a,b]}(x)$ can be approximated by polynomials:
    
    $$\chi_{[a,b]}(x) \approx P_{\text{even}}(x) + P_{\text{odd}}(x)$$
    
    where:
    - $P_{\text{even}}(x) = \frac{1}{2}[\chi(x) + \chi(-x)]$ — symmetric part
    - $P_{\text{odd}}(x) = \frac{1}{2}[\chi(x) - \chi(-x)]$ — anti-symmetric part
    
    ### Why Parity Decomposition?
    
    QSVT naturally implements polynomials of **definite parity**:
    - Even polynomials: $P(-x) = P(x)$
    - Odd polynomials: $P(-x) = -P(x)$
    
    The boxcar function is neither, so we split and run two circuits.
    
    ### Block Encoding
    
    We encode the **position operator** as $\cos(\pi x)$:
    
    $$\langle j | \cos(\pi \hat{X}) | k \rangle = \cos(\pi j / N) \delta_{jk}$$
    
    This maps $x \in [0, 1)$ to eigenvalues $\lambda \in [-1, 1]$.
    
    ### Circuit Flow
    ```
    |f⟩ ──[QSVT_even]──[Measure]── → amplitude_even
    |f⟩ ──[QSVT_odd]──[Measure]──  → amplitude_odd
    
    Total = amplitude_even + amplitude_odd
    ```
    
    ### Advantages
    - ✅ Polynomial approximation can be very accurate
    - ✅ Leverages powerful QSVT framework
    - ❌ Requires solving for QSP angles
    - ❌ Even polynomial extraction has known issues for symmetric intervals
    """)

st.markdown("---")

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
st.header("⚙️ Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Function Selection")
    
    func_type = st.radio(
        "Function Type",
        ["Preset Functions", "Custom Function"],
        horizontal=True
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
        st.markdown(f"**Formula**: {func_formulas.get(func_choice, '')}")
        
    else:
        st.markdown("**Enter a Python expression using `x` as variable:**")
        custom_func = st.text_input(
            "Custom f(x)", 
            value="np.sin(3*np.pi*x) + 1.5",
            help="Use numpy functions with 'np.' prefix. x ranges from 0 to 1."
        )
        func_choice = "custom"
        
        # Validate custom function
        try:
            x_test = np.linspace(0, 1, 10)
            y_test = eval(custom_func)
            if np.any(y_test < 0):
                st.warning("⚠️ Function has negative values. Results may be less accurate.")
            st.success("✅ Function syntax is valid!")
        except Exception as e:
            st.error(f"❌ Invalid function: {e}")
            custom_func = None
    
    # Quantum parameters
    st.markdown("---")
    n_qubits = st.slider("Number of Qubits (n)", 4, 10, 6, 
                         help=f"Grid resolution: 2^n = {2**6} points")
    st.caption(f"📐 Grid: {2**n_qubits} points, Δx = {1/2**n_qubits:.6f}")
    
    shots = st.select_slider("Measurement Shots", 
                             options=[1000, 5000, 10000, 20000, 50000], 
                             value=10000)

with col2:
    st.subheader("🎯 Method & Interval")
    
    method = st.radio(
        "Integration Method", 
        ["Compute-Uncompute (Special Intervals)", 
         "Arithmetic/Comparison (Arbitrary Intervals)",
         "QSVT Parity Decomposition (Arbitrary)"],
        help="Choose the quantum algorithm for integration"
    )
    
    # Method-specific interval selection
    if method == "Compute-Uncompute (Special Intervals)":
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
            
    elif method == "Arithmetic/Comparison (Arbitrary Intervals)":
        st.info("🔧 Uses comparator circuits with multi-controlled gates")
        a_val, b_val = st.slider(
            "Select Interval [a, b]", 0.0, 1.0, (0.25, 0.75), 
            step=0.01, key="arith_slider"
        )
        interval_id = None
        
    else:  # QSVT
        st.info("🌀 Decomposes boxcar into Even + Odd polynomial components")
        a_val, b_val = st.slider(
            "Select Interval [a, b]", 0.0, 1.0, (0.3, 0.7),
            step=0.01
        )
        interval_id = None

# =============================================================================
# FUNCTION VISUALIZATION
# =============================================================================
st.markdown("---")
st.header("📈 Function & Integration Domain Visualization")

# Generate function data for plotting
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

# Create visualization
x_plot, y_plot = get_function_for_plot(func_choice, n_qubits, custom_func)

fig, ax = plt.subplots(figsize=(10, 4))

# Plot full function
ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')

# Highlight integration domain
mask = (x_plot >= a_val) & (x_plot < b_val)
ax.fill_between(x_plot, 0, y_plot, where=mask, alpha=0.3, color='green', 
                label=f'Integration Domain [{a_val:.2f}, {b_val:.2f}]')

# Mark interval boundaries
ax.axvline(x=a_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=b_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Compute exact integral for display
dx = 1.0 / len(x_plot)
exact_integral = np.sum(y_plot[mask]) * dx

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title(f'Function f(x) with Integration Domain D = [{a_val:.2f}, {b_val:.2f}]\n'
             f'Exact Integral ≈ {exact_integral:.4f}', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, max(y_plot) * 1.1)

st.pyplot(fig)
plt.close()

# Show grid points
with st.expander("🔍 View Discretization Grid Points"):
    N = 2**n_qubits
    a_int = int(np.floor(a_val * N))
    b_int = min(int(np.floor(b_val * N)), N-1)
    st.markdown(f"""
    **Grid Information:**
    - Total grid points: N = {N}
    - Interval $[{a_val:.2f}, {b_val:.2f}]$ maps to indices **[{a_int}, {b_int}]**
    - Points in domain: **{b_int - a_int + 1}** out of {N}
    - Grid spacing: Δx = 1/{N} = {1/N:.6f}
    """)

# =============================================================================
# EXECUTION
# =============================================================================
st.markdown("---")
st.header("🚀 Run Quantum Integration")

run_button = st.button("▶️ Execute Quantum Circuit", type="primary", use_container_width=True)

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
                
                st.success("✅ Compute-Uncompute Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Exact Integral", f"{res['integral_exact']:.5f}")
                col2.metric("⚛️ Quantum Estimate", f"{res['integral_est']:.5f}")
                rel_err = abs(res['error']/res['integral_exact'])*100 if res['integral_exact'] != 0 else 0
                col3.metric("📉 Relative Error", f"{rel_err:.2f}%")
                
                st.markdown(f"""
                ### 📋 Results Analysis
                
                | Metric | Value |
                |--------|-------|
                | Circuit Depth | {res['depth_window']} |
                | Gate Count | {res['gate_count_window']} |
                | Overlap $|\\langle\\chi_D|f\\rangle|^2$ | {(res['integral_est']/(res['integral_exact']+1e-10))**2:.4f} |
                
                **Why it works**: The interval state $|\\chi_D\\rangle$ can be prepared with just 
                O(n) = O({n_qubits}) Hadamard gates because the interval aligns with the qubit structure.
                """)
            
            elif method == "Arithmetic/Comparison (Arbitrary Intervals)":
                res = run_arithmetic_integral(n_qubits, effective_func, a_val, b_val, shots)
                
                if func_choice == "custom":
                    x_grid, y_vals = get_function_for_plot("custom", n_qubits, custom_func)
                    N = 2**n_qubits
                    dx = 1/N
                    mask = (x_grid >= a_val) & (x_grid < b_val)
                    res['integral_exact'] = np.sum(y_vals[mask]) * dx
                
                st.success("✅ Arithmetic/Comparison Method Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"📊 Exact [{a_val:.2f}, {b_val:.2f}]", f"{res['integral_exact']:.5f}")
                col2.metric("⚛️ Quantum Estimate", f"{res['integral_est']:.5f}")
                rel_err = abs(res['error']/res['integral_exact'])*100 if res['integral_exact'] != 0 else 0
                col3.metric("📉 Relative Error", f"{rel_err:.1f}%")
                
                st.markdown(f"""
                ### 📋 Algorithm Execution Details
                
                #### Step-by-Step Breakdown
                
                | Step | Operation | Result |
                |------|-----------|--------|
                | 1 | Apply $H^{{\\otimes {n_qubits}}}$ | Uniform superposition over {2**n_qubits} states |
                | 2 | Comparator marking | Mark {res['num_points']} states in {res['interval_int']} |
                | 3 | Apply $U_f^\\dagger$ | Compute overlap with $|f\\rangle$ |
                | 4 | Measure & post-select | Keep ancilla=1 results |
                
                #### Measurement Statistics
                
                | Quantity | Value | Meaning |
                |----------|-------|---------|
                | P(mark=1) | {res['post_select_rate']:.2%} | Post-selection success rate |
                | P(main=0, mark=1) | {res['p_zero_and_marked']:.4f} | Joint probability |
                | $|\\langle\\chi_D|f\\rangle|$ | {res['overlap']:.4f} | Extracted overlap |
                
                #### Circuit Complexity
                - **Depth**: {res['depth']} layers
                - **Gates**: {res['gate_count']} total
                - **MCX gates**: {res['num_points']} (one per marked state)
                """)
                
            else:  # QSVT
                res = run_qsvt_integral_arbitrary(n_qubits, effective_func, a_val, b_val, shots)
                
                if func_choice == "custom":
                    x_grid, y_vals = get_function_for_plot("custom", n_qubits, custom_func)
                    N = 2**n_qubits
                    dx = 1/N
                    mask = (x_grid >= a_val) & (x_grid < b_val)
                    res['integral_exact'] = np.sum(y_vals[mask]) * dx
                
                st.success("✅ QSVT Decomposition Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"📊 Exact [{a_val:.2f}, {b_val:.2f}]", f"{res['integral_exact']:.5f}")
                col2.metric("⚛️ QSVT Estimate", f"{res['integral_est']:.5f}")
                col3.metric("📐 Poly Degrees", f"Even:{res['deg_even']} / Odd:{res['deg_odd']}")
                
                st.markdown(f"""
                ### 📋 Parity Decomposition Results
                
                The boxcar indicator function is split into parity components:
                
                | Component | Value | Polynomial Degree |
                |-----------|-------|-------------------|
                | Even $P_{{even}}(\\lambda)$ | {res['val_even']:.5f} | {res['deg_even']} |
                | Odd $P_{{odd}}(\\lambda)$ | {res['val_odd']:.5f} | {res['deg_odd']} |
                | **Total** | **{res['integral_est']:.5f}** | — |
                
                **Note**: For symmetric intervals around 0.5, the odd component vanishes
                and the even polynomial carries all information.
                """)
                
                # =============================================================
                # QSVT BOXCAR VISUALIZATION (integrated from separate page)
                # =============================================================
                st.markdown("---")
                st.subheader("🔲 QSVT Boxcar Function Visualization")
                
                with st.expander("📈 **View Polynomial Approximation Details**", expanded=True):
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
                        st.markdown("**Boxcar Window Function in Position Space**")
                        
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
                with st.expander("📋 **Discrete Grid Amplitudes**", expanded=False):
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
                            in_interval = "✅" if (x_grid_vis[j] >= a_val and x_grid_vis[j] <= b_val) else "❌"
                            data.append({
                                "Index |j⟩": j,
                                "Position x": f"{x_grid_vis[j]:.4f}",
                                "λ = cos(πx)": f"{lambda_grid[j]:.4f}",
                                "P(λ)": f"{poly_at_grid[j]:.4f}",
                                "In Interval?": in_interval
                            })
                        st.dataframe(data, use_container_width=True)

# =============================================================================
# COMPARISON TABLE
# =============================================================================
st.markdown("---")
with st.expander("📊 **Method Comparison Table**", expanded=False):
    comparison_data = {
        "Method": ["Compute-Uncompute", "Arithmetic/Comparison", "QSVT Parity"],
        "Interval Support": ["Special (half-intervals)", "Arbitrary [a,b]", "Arbitrary [a,b]"],
        "Gate Complexity": ["O(n)", "O(M·n) where M=|D|", "O(d·n) where d=degree"],
        "Accuracy": ["< 1% error", "3-15% error", "Variable (5-30%)"],
        "Post-selection": ["No", "Yes (~M/N rate)", "Yes"],
        "Best Use Case": ["Half-intervals", "Any interval", "Smooth approximations"]
    }
    st.table(pd.DataFrame(comparison_data))

# =============================================================================
# Q&A REFERENCE
# =============================================================================
st.markdown("---")
with st.expander("❓ **Q&A Quick Reference** (For Presentation)", expanded=False):
    st.markdown(r"""
    ### Likely Questions & Answers
    
    **Q: Why use quantum for integration?**
    > Classical: O(N) operations for N grid points. Quantum: Polynomial in n = log(N) qubits 
    > for structured problems. Potential exponential speedup for certain function classes.
    
    **Q: What's the quantum advantage here?**
    > The function is encoded in O(N) amplitudes using only n qubits. Inner products 
    > (integrals) can be estimated with O(1/ε²) measurements regardless of N.
    
    **Q: Why does Compute-Uncompute work so well?**
    > When intervals align with qubit structure, the indicator state |χ_D⟩ has a simple 
    > tensor product form. Only O(n) gates needed vs O(N) classically.
    
    **Q: What are the limitations of the Arithmetic/Comparison method?**
    > Gate count scales with interval size M. Post-selection reduces effective shots.
    > For M ≈ N/2, about 50% of shots are discarded.
    
    **Q: Why split into even/odd for QSVT?**
    > QSVT naturally implements polynomials of definite parity. The boxcar function
    > χ[a,b] has mixed parity, requiring two separate circuits.
    
    **Q: How does error scale with qubits?**
    > More qubits → finer grid → smaller discretization error. 
    > Statistical error from shots: σ ∝ 1/√(shots).
    
    **Q: Can this be run on real quantum hardware?**
    > Yes, but current NISQ devices have limited qubits and high noise.
    > Error mitigation techniques would be needed for practical use.
    """)

st.markdown("---")
st.caption("💡 **Tip**: Expand the theory sections above for detailed explanations during Q&A!")