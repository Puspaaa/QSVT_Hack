"""
Shared interactive components for the QSVT presentation slides.
Provides visual widgets: Bloch sphere, matrix heatmaps, circuit diagrams,
probability bar charts, polynomial plots, and reference citation boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st


# ── Presentation-wide CSS ────────────────────────────────────────────────────
PRESENTATION_CSS = """
<style>
    /* Larger header fonts for readability from a distance */
    .main h1 { font-size: 2.8rem !important; }
    .main h2 { font-size: 2.0rem !important; }
    .main h3 { font-size: 1.5rem !important; }

    /* Slide content area */
    .slide-content { padding: 0.5rem 1rem; }

    /* Reference citation box */
    .ref-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4a90d9;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        border-radius: 0 6px 6px 0;
    }
    .ref-box .ref-tag {
        font-weight: 700;
        color: #4a90d9;
    }

    /* Navigation bar */
    .nav-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #262730;
        padding: 0.4rem 2rem;
        z-index: 999;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Slide progress bar */
    .progress-outer {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        width: 100%;
        margin: 0.25rem 0;
    }
    .progress-inner {
        height: 6px;
        background: linear-gradient(90deg, #4a90d9, #7b61ff);
        border-radius: 3px;
    }

    /* Key concept highlight box — targets the bordered st.container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 100%) !important;
        border: 1px solid #c4d7f2 !important;
        border-radius: 10px !important;
        padding: 0.4rem 0.6rem !important;
        margin: 1rem 0 !important;
        font-size: 1.05rem;
    }
</style>
"""


def inject_css():
    """Inject presentation CSS into the page. Call once per page."""
    st.markdown(PRESENTATION_CSS, unsafe_allow_html=True)


# ── Reference citation ───────────────────────────────────────────────────────

# Master reference database
REFERENCES = {
    "NC2000": {
        "authors": "M. A. Nielsen & I. L. Chuang",
        "title": "Quantum Computation and Quantum Information",
        "venue": "Cambridge University Press",
        "year": 2000,
        "url": "https://doi.org/10.1017/CBO9780511976667",
    },
    "Gilyen2019": {
        "authors": "A. Gilyén, Y. Su, G. H. Low, N. Wiebe",
        "title": "Quantum singular value transformations and beyond: exponential improvements for quantum matrix arithmetics",
        "venue": "STOC 2019",
        "year": 2019,
        "url": "https://doi.org/10.1145/3313276.3316366",
    },
    "Martyn2021": {
        "authors": "J. M. Martyn, Z. M. Rossi, A. K. Tan, I. L. Chuang",
        "title": "Grand Unification of Quantum Algorithms",
        "venue": "PRX Quantum 2, 040203",
        "year": 2021,
        "url": "https://doi.org/10.1103/PRXQuantum.2.040203",
    },
    "Low2017": {
        "authors": "G. H. Low & I. L. Chuang",
        "title": "Optimal Hamiltonian Simulation by Quantum Signal Processing",
        "venue": "Physical Review Letters 118, 010501",
        "year": 2017,
        "url": "https://doi.org/10.1103/PhysRevLett.118.010501",
    },
    "Low2019": {
        "authors": "G. H. Low & I. L. Chuang",
        "title": "Hamiltonian Simulation by Qubitization",
        "venue": "Quantum 3, 163",
        "year": 2019,
        "url": "https://doi.org/10.22331/q-2019-07-12-163",
    },
    "Berry2015": {
        "authors": "D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, R. D. Somma",
        "title": "Simulating Hamiltonian dynamics with a truncated Taylor series",
        "venue": "Physical Review Letters 114, 090502",
        "year": 2015,
        "url": "https://doi.org/10.1103/PhysRevLett.114.090502",
    },
    "Dong2021": {
        "authors": "Y. Dong, X. Meng, K. B. Whaley, L. Lin",
        "title": "Efficient Phase-Factor Evaluation in Quantum Signal Processing",
        "venue": "Physical Review A 103, 042419",
        "year": 2021,
        "url": "https://doi.org/10.1103/PhysRevA.103.042419",
    },
    "Brassard2002": {
        "authors": "G. Brassard, P. Hoyer, M. Mosca, A. Tapp",
        "title": "Quantum Amplitude Amplification and Estimation",
        "venue": "Contemporary Mathematics 305, pp. 53-74",
        "year": 2002,
        "url": "https://doi.org/10.1090/conm/305/05215",
    },
    "Costa2019": {
        "authors": "P. C. S. Costa, S. Jordan, A. Ostrander",
        "title": "Quantum algorithm for simulating the wave equation",
        "venue": "Physical Review A 99, 012323",
        "year": 2019,
        "url": "https://doi.org/10.1103/PhysRevA.99.012323",
    },
}


def reference(tag: str):
    """Render a styled citation box for a given reference tag."""
    ref = REFERENCES.get(tag)
    if ref is None:
        st.warning(f"Unknown reference tag: {tag}")
        return
    html = (
        f'<div class="ref-box">'
        f'<span class="ref-tag">[{tag}]</span> '
        f'{ref["authors"]}, '
        f'<em>"{ref["title"]}"</em>, '
        f'{ref["venue"]} ({ref["year"]}). '
        f'<a href="{ref["url"]}" target="_blank">link</a>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def reference_list(tags: list[str]):
    """Render a compact list of references."""
    for tag in tags:
        reference(tag)


# ── Key concept box ──────────────────────────────────────────────────────────

def key_concept(text: str):
    """Render a highlighted key-concept box (supports markdown/LaTeX).

    Converts HTML bold/italic to Markdown equivalents so the text can be
    rendered by Streamlit's native markdown engine (which handles $…$ LaTeX)
    inside a bordered container.
    """
    import re

    # Convert HTML formatting tags to Markdown equivalents
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text)
    text = re.sub(r'<br\s*/?>', '  \n', text)

    with st.container(border=True):
        st.markdown(text)


# ── Navigation helpers ───────────────────────────────────────────────────────

def slide_header(title: str, subtitle: str = ""):
    """Render a slide title with optional subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def progress_bar(current: int, total: int):
    """Render a thin progress bar."""
    pct = int(100 * (current + 1) / total)
    st.markdown(
        f'<div class="progress-outer">'
        f'<div class="progress-inner" style="width:{pct}%"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Slide {current + 1} / {total}")


# ── Bloch sphere ─────────────────────────────────────────────────────────────

def bloch_sphere(theta: float, phi: float, figsize=(4, 4)):
    """
    Draw a Bloch sphere with a state vector at (theta, phi).
    theta: polar angle from |0⟩  (0 = |0⟩, π = |1⟩)
    phi: azimuthal phase
    Returns a matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='lightblue', alpha=0.15, linewidth=0.4)

    # Axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], 'k-', alpha=0.2, linewidth=0.5)

    # Labels
    ax.text(0, 0, 1.45, r'$|0\rangle$', fontsize=12, ha='center')
    ax.text(0, 0, -1.45, r'$|1\rangle$', fontsize=12, ha='center')
    ax.text(1.45, 0, 0, r'$|+\rangle$', fontsize=10, ha='center')
    ax.text(-1.45, 0, 0, r'$|-\rangle$', fontsize=10, ha='center')

    # State vector
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)
    ax.quiver(0, 0, 0, sx, sy, sz, color='#7b61ff', arrow_length_ratio=0.12, linewidth=2.5)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ── Matrix heatmap ───────────────────────────────────────────────────────────

def matrix_heatmap(matrix, title="", figsize=(5, 4), annotate=True, cmap="RdBu_r"):
    """Color-coded matrix visualization."""
    fig, ax = plt.subplots(figsize=figsize)
    n = matrix.shape[0]
    vmax = np.max(np.abs(matrix))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(np.real(matrix), cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    if annotate and n <= 8:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if np.iscomplex(val) and np.abs(np.imag(val)) > 1e-10:
                    txt = f"{val:.2f}"
                else:
                    txt = f"{np.real(val):.2f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=max(7, 12 - n))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


# ── Probability bar chart ────────────────────────────────────────────────────

def probability_bar_chart(probs, labels=None, title="Measurement Probabilities",
                          figsize=(6, 3), color="#4a90d9"):
    """Bar chart of measurement probabilities."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(probs)
    if labels is None:
        labels = [f"|{i}⟩" for i in range(n)]
    bars = ax.bar(range(n), probs, color=color, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=max(8, 13 - n))
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(probs) * 1.2 + 0.01)
    ax.set_title(title, fontsize=12, fontweight='bold')
    # Value labels
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{p:.3f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    return fig


# ── Simple circuit diagram (matplotlib-based) ────────────────────────────────

def draw_circuit_diagram(n_qubits, gates, figsize=(10, None), title=""):
    """
    Draw a simple quantum circuit diagram.
    
    gates: list of dicts, each with:
        - 'name': str (gate label)
        - 'qubits': list[int] (target qubits, 0-indexed from top)
        - 'style': 'box' | 'circle' | 'barrier' (default: 'box')
        - 'color': str (optional, default '#4a90d9')
    """
    if figsize[1] is None:
        figsize = (figsize[0], max(2, n_qubits * 0.7))
    fig, ax = plt.subplots(figsize=figsize)

    n_gates = len(gates)
    x_margin = 0.5
    gate_spacing = 1.2
    wire_len = x_margin + (n_gates + 1) * gate_spacing

    # Draw wires
    for q in range(n_qubits):
        y = -q
        ax.plot([0, wire_len], [y, y], 'k-', linewidth=1, zorder=0)
        ax.text(-0.3, y, f"|q{q}⟩", ha='right', va='center', fontsize=11)

    # Draw gates
    for gi, gate in enumerate(gates):
        x = x_margin + (gi + 1) * gate_spacing
        qubits = gate['qubits']
        style = gate.get('style', 'box')
        color = gate.get('color', '#4a90d9')
        name = gate['name']

        if style == 'barrier':
            for q in range(n_qubits):
                ax.plot([x, x], [-q - 0.3, -q + 0.3], 'k--', linewidth=1, alpha=0.5)
            continue

        if len(qubits) == 2:
            # Control-target line
            ax.plot([x, x], [-qubits[0], -qubits[1]], 'k-', linewidth=1.5, zorder=1)
            # Control dot
            ax.plot(x, -qubits[0], 'ko', markersize=8, zorder=2)
            # Target
            rect = patches.FancyBboxPatch((x - 0.3, -qubits[1] - 0.3), 0.6, 0.6,
                                           boxstyle="round,pad=0.05", facecolor=color,
                                           edgecolor='black', linewidth=1.2, zorder=2)
            ax.add_patch(rect)
            ax.text(x, -qubits[1], name, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white', zorder=3)
        else:
            for q in qubits:
                rect = patches.FancyBboxPatch((x - 0.3, -q - 0.3), 0.6, 0.6,
                                               boxstyle="round,pad=0.05", facecolor=color,
                                               edgecolor='black', linewidth=1.2, zorder=2)
                ax.add_patch(rect)
                ax.text(x, -q, name, ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white', zorder=3)

    # Measurement symbols
    ax.set_xlim(-0.8, wire_len + 0.3)
    ax.set_ylim(-n_qubits + 0.5, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Polynomial / Chebyshev plot ──────────────────────────────────────────────

def polynomial_plot(target_fn, approx_coeffs_or_fn, domain=(-1, 1), n_points=300,
                    target_label="Target f(x)", approx_label="Polynomial P(x)",
                    title="Polynomial Approximation", figsize=(7, 4)):
    """
    Plot a target function vs. its polynomial approximation.
    approx_coeffs_or_fn: either Chebyshev coeff array or a callable.
    """
    from numpy.polynomial.chebyshev import chebval
    x = np.linspace(domain[0], domain[1], n_points)
    y_target = np.array([target_fn(xi) for xi in x])

    if callable(approx_coeffs_or_fn):
        y_approx = np.array([approx_coeffs_or_fn(xi) for xi in x])
    else:
        y_approx = chebval(x, approx_coeffs_or_fn)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                                     gridspec_kw={'hspace': 0.05})

    ax1.plot(x, y_target, 'b-', linewidth=2, label=target_label)
    ax1.plot(x, y_approx, 'r--', linewidth=2, label=approx_label)
    ax1.set_ylabel("f(x)")
    ax1.legend(fontsize=10)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.set_xlim(domain)
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.3)

    # Error subplot
    error = np.abs(y_target - y_approx)
    ax2.semilogy(x, error + 1e-16, 'green', linewidth=1.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("|Error|")
    ax2.set_xlim(domain)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── QSVT circuit schematic ──────────────────────────────────────────────────

def qsvt_circuit_schematic(degree, figsize=(12, 2.5)):
    """Draw a stylized QSVT circuit: alternating U_A and Rz(phi) blocks."""
    fig, ax = plt.subplots(figsize=figsize)

    # Signal wire
    y_sig = 1.0
    y_sys = 0.0
    x_start = 0.0
    spacing = 1.8
    show_gates = min(degree, 5)  # Show at most 5 blocks, then ellipsis

    total_width = (show_gates + 2) * spacing + 2
    ax.plot([x_start - 0.5, total_width], [y_sig, y_sig], 'k-', linewidth=1.2)
    ax.plot([x_start - 0.5, total_width], [y_sys, y_sys], 'k-', linewidth=1.2)

    # Labels
    ax.text(-1.0, y_sig, "signal", ha='right', va='center', fontsize=10, style='italic')
    ax.text(-1.0, y_sys, "system", ha='right', va='center', fontsize=10, style='italic')

    x = x_start

    # H gate
    rect = patches.FancyBboxPatch((x - 0.25, y_sig - 0.25), 0.5, 0.5,
                                   boxstyle="round,pad=0.05", facecolor='#f5a623',
                                   edgecolor='black', linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y_sig, "H", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)

    x += spacing * 0.8

    # Initial Rz
    rect = patches.FancyBboxPatch((x - 0.3, y_sig - 0.25), 0.6, 0.5,
                                   boxstyle="round,pad=0.05", facecolor='#7b61ff',
                                   edgecolor='black', linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y_sig, r"$R_z(\phi_0)$", ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=3)

    x += spacing * 0.8

    for i in range(show_gates):
        # U_A block (spans both wires)
        rect = patches.FancyBboxPatch((x - 0.35, y_sys - 0.3), 0.7, y_sig - y_sys + 0.6,
                                       boxstyle="round,pad=0.08", facecolor='#4a90d9',
                                       edgecolor='black', linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        label = r"$U_A$" if (i % 2 == 0) else r"$U_A^\dagger$"
        ax.text(x, (y_sig + y_sys) / 2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=3)

        x += spacing * 0.7

        # Rz on signal (controlled by ancilla success)
        rect = patches.FancyBboxPatch((x - 0.3, y_sig - 0.25), 0.6, 0.5,
                                       boxstyle="round,pad=0.05", facecolor='#7b61ff',
                                       edgecolor='black', linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y_sig, f"$\\phi_{{{i+1}}}$", ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=3)

        # Connection line to system
        ax.plot([x, x], [y_sys + 0.05, y_sig - 0.25], 'k-', linewidth=0.8, alpha=0.4)

        x += spacing * 0.7

        if i == 2 and show_gates > 4:
            ax.text(x, (y_sig + y_sys) / 2, "⋯", fontsize=20, ha='center', va='center')
            x += spacing * 0.5

    # Final H gate
    rect = patches.FancyBboxPatch((x - 0.25, y_sig - 0.25), 0.5, 0.5,
                                   boxstyle="round,pad=0.05", facecolor='#f5a623',
                                   edgecolor='black', linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y_sig, "H", ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)

    ax.set_xlim(-1.5, total_width + 0.5)
    ax.set_ylim(-0.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"QSVT Circuit — degree {degree} polynomial", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig
