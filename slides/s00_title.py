"""Slide 0: Title Slide"""

import streamlit as st
from slides.components import slide_header, inject_css, key_concept

TITLE = "Title & Motivation"


def render():
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align:center; font-size:3.2rem;'>"
        "Quantum Singular Value Transformation<br>"
        "<span style='font-size:1.6rem; color:#666;'>for Scientific Computing</span>"
        "</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; font-size:1.15rem; color:#888;'>"
        "Solving PDEs &amp; Computing Integrals with Polynomial Quantum Algorithms"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Teaser columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            #### The Problem
            Classical PDE solvers scale as $O(N)$ per step.  
            Can quantum computers do better?
            """
        )
    with col2:
        st.markdown(
            """
            #### The Framework
            **QSVT** — a single framework that unifies  
            most known quantum algorithms.
            """
        )
    with col3:
        st.markdown(
            """
            #### This Work
            1D & 2D advection-diffusion + numerical integration,  
            all implemented & simulated.
            """
        )

    st.markdown("---")

    key_concept(
        "<b>Key promise:</b> Encode $N = 2^n$ grid points in just $n$ qubits, "
        "then apply matrix functions in $O(\\text{poly}(n))$ gates — exponentially fewer resources."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#999; font-size:0.9rem;'>"
        "QHack 2026 — Quantum Algorithms for Scientific Computing"
        "</p>",
        unsafe_allow_html=True,
    )
