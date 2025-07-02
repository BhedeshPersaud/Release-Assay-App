import streamlit as st
import pickle
import os

def ensure_state_keys(keys):
    for key in keys:
        if key == "stored_gel_data":
            st.write("🔁 Checking for stored_gel_data.pkl")
            file_path = os.path.join(os.getcwd(), "stored_gel_data.pkl")
            st.write(f"Looking at: {file_path}")
            if os.path.exists(file_path):
                st.write("📂 Found stored_gel_data.pkl, loading...")
                try:
                    with open(file_path, "rb") as f:
                        st.session_state["stored_gel_data"] = pickle.load(f)
                        st.write("✅ Loaded stored_gel_data:", st.session_state["stored_gel_data"])
                except Exception as e:
                    st.warning(f"⚠️ Failed to load stored gel data: {e}")
                    st.session_state["stored_gel_data"] = {}
            else:
                st.write("❌ stored_gel_data.pkl not found, starting fresh")
                st.session_state["stored_gel_data"] = {}

