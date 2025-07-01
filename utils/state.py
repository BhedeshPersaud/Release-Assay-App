import streamlit as st
import pickle
import os

def ensure_state_keys(keys):
    for key in keys:
        if key not in st.session_state:
            if key == "stored_gel_data":
                if os.path.exists("stored_gel_data.pkl"):
                    try:
                        with open("stored_gel_data.pkl", "rb") as f:
                            st.session_state["stored_gel_data"] = pickle.load(f)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load stored gel data: {e}")
                        st.session_state["stored_gel_data"] = {}
                else:
                    st.session_state["stored_gel_data"] = {}
            else:
                st.session_state[key] = None
