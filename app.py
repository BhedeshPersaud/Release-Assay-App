import streamlit as st
from utils.state import ensure_state_keys

# -------------------  Page configuration -----------------------------
st.set_page_config(
    page_title="Dye Release App",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# ------------------- Initialize persistent state keys ------------------
ensure_state_keys(["stored_gel_data"])

# ------------------- Render welcome page for user ------------------------
import pages._1_welcome

