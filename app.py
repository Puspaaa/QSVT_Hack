import streamlit as st
from slides.components import inject_css, progress_bar

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QSVT for Scientific Computing — Presentation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

# ── Import all slide modules ─────────────────────────────────────────────────
from slides import (
    s00_title,
    s01_quantum_circuits,
    s02_amplitude_encoding,
    s03_block_encoding,
    s04_qsvt,
    s05_hamiltonian_sim,
    s06_qsvt_for_pdes,
    s06b_demo_1d_pde,
    s07_qsvt_for_integration,
    s07b_demo_2d_pde,
    s08_our_approach,
    s08b_demo_integration,
    s09_summary,
)

SLIDES = [
    s00_title,
    s01_quantum_circuits,
    s02_amplitude_encoding,
    s03_block_encoding,
    s04_qsvt,
    s05_hamiltonian_sim,
    s06_qsvt_for_pdes,
    s06b_demo_1d_pde,
    s07_qsvt_for_integration,
    s07b_demo_2d_pde,
    s08_our_approach,
    s08b_demo_integration,
    s09_summary,
]

TOTAL = len(SLIDES)

# ── Session state for slide index ────────────────────────────────────────────
if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0


def go_prev():
    if st.session_state.slide_idx > 0:
        st.session_state.slide_idx -= 1


def go_next():
    if st.session_state.slide_idx < TOTAL - 1:
        st.session_state.slide_idx += 1


def jump_to(idx):
    st.session_state.slide_idx = idx


# ── Sidebar: slide outline ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Presentation Outline")
    for i, slide_mod in enumerate(SLIDES):
        label = f"{'→ ' if i == st.session_state.slide_idx else '   '}{i}. {slide_mod.TITLE}"
        if st.button(label, key=f"nav_{i}", use_container_width=True):
            jump_to(i)
            st.rerun()
    st.markdown("---")

# ── Render current slide ─────────────────────────────────────────────────────
idx = st.session_state.slide_idx
current_slide = SLIDES[idx]

# Render content
current_slide.render()

# ── Navigation bar ───────────────────────────────────────────────────────────
st.markdown("---")
progress_bar(idx, TOTAL)

nav_left, nav_center, nav_right = st.columns([1, 2, 1])

with nav_left:
    st.button("← Previous", on_click=go_prev, disabled=(idx == 0), use_container_width=True)

with nav_center:
    st.markdown(
        f"<p style='text-align:center; color:#888; margin-top:0.5rem;'>"
        f"<b>{current_slide.TITLE}</b></p>",
        unsafe_allow_html=True,
    )

with nav_right:
    st.button("Next →", on_click=go_next, disabled=(idx == TOTAL - 1), use_container_width=True)