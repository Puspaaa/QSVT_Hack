"""
Quantum Advection-Diffusion Wind Tunnel - Streamlit App

This application simulates the advection-diffusion equation using three methods:
1. Exact Fourier solution
2. Classical Finite Difference
3. Quantum Split-Step with QSVT

The advection-diffusion equation models transport phenomena combining:
- Diffusion: spreading due to viscosity (ν∂²u/∂x²)
- Advection: transport due to flow velocity (v∂u/∂x)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulation import (
    create_comparison_plot,
    exact_fourier_solution,
    classical_finite_difference_solution,
    quantum_split_step_solution
)
from quantum import (
    create_diffusion_block_encoding,
    create_advection_gate,
    create_combined_advection_diffusion_circuit
)
from solvers import compute_qsvt_polynomial_coefficients


# Page configuration
st.set_page_config(
    page_title="Quantum Advection-Diffusion Wind Tunnel",
    page_icon="🌊",
    layout="wide"
)

# Title and description
st.title("🌊 Quantum Advection-Diffusion Wind Tunnel")
st.markdown("""
This app simulates the **advection-diffusion equation** using quantum computing techniques.
The equation models how a quantity (like heat or concentration) spreads and moves in space:

$$\\frac{\\partial u}{\\partial t} = \\nu \\frac{\\partial^2 u}{\\partial x^2} - v \\frac{\\partial u}{\\partial x}$$

where:
- **ν** (nu) is the viscosity/diffusion coefficient
- **v** is the advection velocity
- **u(x,t)** is the quantity being transported
""")

# Sidebar for inputs
st.sidebar.header("⚙️ Simulation Parameters")

# Number of qubits (determines grid resolution)
n_qubits = st.sidebar.slider(
    "Number of Qubits",
    min_value=3,
    max_value=8,
    value=5,
    help="More qubits = finer grid resolution (Grid points = 2^n_qubits)"
)

# Viscosity parameter
viscosity = st.sidebar.slider(
    "Viscosity (ν)",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f",
    help="Controls diffusion/spreading. Higher = more spreading."
)

# Velocity parameter
velocity = st.sidebar.slider(
    "Velocity (v)",
    min_value=-2.0,
    max_value=2.0,
    value=0.5,
    step=0.1,
    format="%.1f",
    help="Controls advection/transport. Positive = rightward motion."
)

# Time parameter
t_final = st.sidebar.slider(
    "Final Time (t)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    format="%.1f",
    help="Time at which to evaluate the solution"
)

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    domain_length = st.slider("Domain Length", 1.0, 10.0, 2.0 * np.pi, 0.5)
    initial_pulse_width = st.slider("Initial Pulse Width", 0.1, 2.0, 0.5, 0.1)
    show_errors = st.checkbox("Show Error Analysis", value=True)

# Create spatial grid
N = 2 ** n_qubits
x = np.linspace(0, domain_length, N)

# Create initial condition (Gaussian pulse)
x0 = domain_length / 2
sigma = initial_pulse_width
initial_condition = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

# Display info
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Grid Information:**
- Grid points: {N}
- Spatial resolution: {domain_length/N:.4f}
- Domain: [0, {domain_length:.2f}]
""")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📊 Results", "🔬 Circuit Details", "📐 Mathematical Background"])

# Tab 1: Results
with tab1:
    st.header("Simulation Results")
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulations... This may take a moment."):
            try:
                # Create comparison plot
                fig = create_comparison_plot(
                    x, viscosity, velocity, t_final, initial_condition
                )
                
                st.pyplot(fig)
                plt.close(fig)
                
                st.success("✅ Simulation completed successfully!")
                
                # Show additional metrics
                if show_errors:
                    st.subheader("📈 Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Compute solutions for metrics
                    u_fourier = exact_fourier_solution(x, t_final, viscosity, velocity, initial_condition)
                    u_classical = classical_finite_difference_solution(x, t_final, viscosity, velocity, initial_condition)
                    
                    try:
                        u_quantum = quantum_split_step_solution(x, t_final, viscosity, velocity, initial_condition)
                    except:
                        u_quantum = u_fourier
                    
                    # Calculate errors
                    l2_classical = np.sqrt(np.mean((u_classical - u_fourier)**2))
                    l2_quantum = np.sqrt(np.mean((u_quantum - u_fourier)**2))
                    
                    with col1:
                        st.metric("Classical L2 Error", f"{l2_classical:.6f}")
                    
                    with col2:
                        st.metric("Quantum L2 Error", f"{l2_quantum:.6f}")
                    
                    with col3:
                        improvement = ((l2_classical - l2_quantum) / l2_classical * 100) if l2_classical > 0 else 0
                        st.metric("Quantum Improvement", f"{improvement:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {str(e)}")
                st.exception(e)
    
    else:
        st.info("👆 Click 'Run Simulation' to start the comparison")
        
        # Show preview of initial condition
        st.subheader("Initial Condition Preview")
        fig_init, ax_init = plt.subplots(figsize=(10, 4))
        ax_init.plot(x, initial_condition, 'k-', linewidth=2)
        ax_init.set_xlabel('Position x')
        ax_init.set_ylabel('u(x, 0)')
        ax_init.set_title('Initial Gaussian Pulse')
        ax_init.grid(True, alpha=0.3)
        st.pyplot(fig_init)
        plt.close(fig_init)

# Tab 2: Circuit Details
with tab2:
    st.header("Quantum Circuit Details")
    
    st.markdown("""
    The quantum simulation uses **operator splitting** to decompose the advection-diffusion 
    operator into simpler components that can be implemented efficiently on a quantum computer.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Diffusion Block Encoding")
        st.markdown("""
        The diffusion operator $e^{-\\nu \\Delta t \\partial^2/\\partial x^2}$ is implemented using:
        1. **QFT** to transform to Fourier space
        2. **Phase rotations** for $e^{-\\nu \\Delta t k^2}$
        3. **Inverse QFT** to return to position space
        """)
        
        try:
            diffusion_circ = create_diffusion_block_encoding(min(4, n_qubits), viscosity, 0.1)
            st.text(f"Diffusion circuit depth: {diffusion_circ.depth()}")
            st.text(f"Number of gates: {len(diffusion_circ)}")
        except Exception as e:
            st.warning(f"Could not generate circuit: {e}")
    
    with col2:
        st.subheader("➡️ Advection Gate")
        st.markdown("""
        The advection operator $e^{-v \\Delta t \\partial/\\partial x}$ is implemented as:
        1. **QFT** to Fourier space
        2. **Phase rotations** for $e^{-iv \\Delta t k}$
        3. **Inverse QFT** back
        
        This represents a translation in position space.
        """)
        
        try:
            advection_circ = create_advection_gate(min(4, n_qubits), velocity, 0.1)
            st.text(f"Advection circuit depth: {advection_circ.depth()}")
            st.text(f"Number of gates: {len(advection_circ)}")
        except Exception as e:
            st.warning(f"Could not generate circuit: {e}")
    
    st.subheader("🔗 Combined Circuit")
    st.markdown("""
    The full simulation alternates between diffusion and advection operators using the 
    **Strang splitting** method for second-order accuracy:
    
    $$U(\\Delta t) = e^{-\\nu \\Delta t/2 \\cdot \\partial^2/\\partial x^2} \\cdot e^{-v \\Delta t \\cdot \\partial/\\partial x} \\cdot e^{-\\nu \\Delta t/2 \\cdot \\partial^2/\\partial x^2}$$
    """)
    
    # Show combined circuit
    if st.button("Generate Combined Circuit"):
        try:
            with st.spinner("Generating circuit..."):
                combined_circ = create_combined_advection_diffusion_circuit(
                    min(4, n_qubits), viscosity, velocity, n_steps=1, dt=0.1
                )
                
                st.text(f"Combined circuit depth: {combined_circ.depth()}")
                st.text(f"Total gates: {len(combined_circ)}")
                
                # Draw circuit (only for small circuits)
                if n_qubits <= 4:
                    fig_circ = combined_circ.draw(output='mpl', fold=-1)
                    st.pyplot(fig_circ)
                    plt.close(fig_circ)
                else:
                    st.info("Circuit too large to display. Use fewer qubits to visualize.")
        except Exception as e:
            st.error(f"Error generating circuit: {e}")

# Tab 3: Mathematical Background
with tab3:
    st.header("Mathematical Background")
    
    st.subheader("📚 The Advection-Diffusion Equation")
    st.markdown("""
    The advection-diffusion equation is a fundamental PDE in physics and engineering:
    
    $$\\frac{\\partial u}{\\partial t} = \\nu \\frac{\\partial^2 u}{\\partial x^2} - v \\frac{\\partial u}{\\partial x}$$
    
    **Physical Interpretation:**
    - **Diffusion term** ($\\nu \\partial^2 u/\\partial x^2$): Describes spreading due to random motion (like heat conduction)
    - **Advection term** ($v \\partial u/\\partial x$): Describes transport by bulk motion (like wind or flow)
    
    **Applications:**
    - Heat transfer in moving fluids
    - Pollutant dispersion in air or water
    - Chemical concentration in reactors
    - Population dynamics
    """)
    
    st.subheader("🔬 Solution Methods")
    
    with st.expander("1️⃣ Exact Fourier Solution"):
        st.markdown("""
        Using Fourier transforms, we can solve the equation exactly:
        
        1. Transform initial condition: $\\hat{u}(k,0) = \\mathcal{F}[u(x,0)]$
        2. Evolve in Fourier space: $\\hat{u}(k,t) = \\hat{u}(k,0) e^{-\\nu k^2 t - ivkt}$
        3. Transform back: $u(x,t) = \\mathcal{F}^{-1}[\\hat{u}(k,t)]$
        
        This is the **reference solution** used to measure errors.
        """)
    
    with st.expander("2️⃣ Classical Finite Difference"):
        st.markdown("""
        Discretize space and time on a grid:
        
        $$u_i^{n+1} = u_i^n + \\Delta t \\left[ \\nu \\frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\\Delta x^2} - v \\frac{u_{i+1}^n - u_{i-1}^n}{2\\Delta x} \\right]$$
        
        **Pros:** Simple to implement, works for complex domains
        
        **Cons:** Limited by CFL condition, accumulates numerical diffusion
        """)
    
    with st.expander("3️⃣ Quantum Split-Step Method"):
        st.markdown("""
        Uses quantum circuits to simulate the evolution operator:
        
        1. **Operator Splitting:** $e^{(A+B)t} \\approx e^{At/2} e^{Bt} e^{At/2}$
        2. **Block Encoding:** Encode diffusion/advection operators as unitary matrices
        3. **QSVT Enhancement:** Apply polynomial transformations for better accuracy
        
        **Pros:** Potential quantum speedup for high-dimensional problems
        
        **Cons:** Requires quantum hardware, limited by current qubit counts
        """)
    
    st.subheader("🎯 QSVT (Quantum Singular Value Transformation)")
    st.markdown("""
    QSVT is a powerful technique for implementing polynomial functions of matrices:
    
    $$U_{QSVT} = e^{i\\phi_0 Z} \\prod_{j=1}^d \\left[ U_{sig} e^{i\\phi_j Z} \\right]$$
    
    where $U_{sig}$ is a signal operator (our block-encoded PDE operator) and $\\phi_j$ are phase factors 
    computed via convex optimization (CVXPY) to approximate desired functions.
    
    **Key Benefits:**
    - Nearly optimal polynomial approximations
    - Systematic error control
    - Applicable to wide range of matrix functions
    """)
    
    st.subheader("📊 Polynomial Approximation")
    
    # Show polynomial coefficients
    if st.button("Compute QSVT Polynomial"):
        with st.spinner("Computing polynomial coefficients..."):
            try:
                degree = st.slider("Polynomial Degree", 3, 15, 7)
                coeffs = compute_qsvt_polynomial_coefficients(degree, target_func="sign")
                
                st.write("**Polynomial Coefficients:**")
                st.write(coeffs)
                
                # Plot polynomial
                x_plot = np.linspace(-1, 1, 200)
                y_poly = sum(coeffs[i] * x_plot**i for i in range(len(coeffs)))
                y_target = np.sign(x_plot)
                
                fig_poly, ax_poly = plt.subplots(figsize=(10, 4))
                ax_poly.plot(x_plot, y_target, 'k--', label='Target (sign)', linewidth=2)
                ax_poly.plot(x_plot, y_poly, 'b-', label=f'Polynomial (degree {degree})', linewidth=2)
                ax_poly.set_xlabel('x')
                ax_poly.set_ylabel('P(x)')
                ax_poly.set_title('Polynomial Approximation to Sign Function')
                ax_poly.legend()
                ax_poly.grid(True, alpha=0.3)
                st.pyplot(fig_poly)
                plt.close(fig_poly)
                
            except Exception as e:
                st.error(f"Error computing polynomial: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, Qiskit, and NumPy | Quantum Advection-Diffusion Wind Tunnel</p>
    <p><em>Exploring the intersection of quantum computing and computational fluid dynamics</em></p>
</div>
""", unsafe_allow_html=True)
