import streamlit as st
import pickle
import os
from utils.state import ensure_state_keys

ensure_state_keys(["stored_gel_data"])


# ✅ Load data from disk if available
if "stored_gel_data" not in st.session_state:
    if os.path.exists("stored_gel_data.pkl"):
        with open("stored_gel_data.pkl", "rb") as f:
            st.session_state["stored_gel_data"] = pickle.load(f)
    else:
        st.session_state["stored_gel_data"] = {}

def render():
    st.set_page_config(
        page_title="Dye Release App",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🔬 Welcome to the Dye Release Analysis App")

    st.markdown("""
    This app helps you analyze 96-well plate reader data from dye release experiments.

    Use the **sidebar** to navigate:
    - **Calibration Curve**: Build a standard curve from known concentrations.
    - **Release Profile**: Generate dye release profiles from experiment data.
    - **Database**: Store and access your release tables.

    More features coming soon!
    """)

render()
