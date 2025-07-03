import streamlit as st
import pickle
import os

def ensure_state_keys(keys):
    for key in keys:
        if key == "stored_gel_data":
            file_path = os.path.join(os.getcwd(), "stored_gel_data.pkl")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        st.session_state["stored_gel_data"] = pickle.load(f)
                except Exception as e:
                    st.warning(f"⚠️ Failed to load stored gel data: {e}")
                    st.session_state["stored_gel_data"] = {}
            else:
                st.session_state["stored_gel_data"] = {}


