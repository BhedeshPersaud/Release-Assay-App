import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from utils.state import ensure_state_keys


if "stored_gel_data" not in st.session_state or st.session_state["stored_gel_data"] is None:
    st.session_state["stored_gel_data"] = {}

from utils.state import ensure_state_keys
ensure_state_keys(["stored_gel_data"])

def render():

    st.title("📁 Saved Gel Database")

    if "stored_gel_data" not in st.session_state or not st.session_state["stored_gel_data"]:
        st.info("No gel data has been stored yet.")
        return

    gels_to_delete = []

    for gel_name, data in st.session_state["stored_gel_data"].items():
        with st.expander(f"🧪 {gel_name}", expanded=False):
            col1, col2 = st.columns([5, 1])
            with col2:
                if st.button("❌ Delete", key=f"delete_{gel_name}"):
                    gels_to_delete.append(gel_name)

            st.markdown("### 📊 Release Profile Table")
            if isinstance(data, pd.DataFrame):
                st.warning(f"⚠️ Skipping {gel_name} — saved in an older format.")
                continue

            st.dataframe(data.get("release_df", pd.DataFrame()), use_container_width=True)

            # CSV Export Button
            csv = data["release_df"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Release Table as CSV",
                data=csv,
                file_name=f"{gel_name}_release_profile.csv",
                mime="text/csv",
                key=f"download_csv_{gel_name}"
            )

            st.markdown("### 📈 Model Fits")
            model_fits = data.get("model_fits", {})
            if not isinstance(model_fits, dict):
                st.warning(f"⚠️ No model fits found for {gel_name} due to format mismatch.")
                continue

            if model_fits:
                fit_lines = []
                for model_name, model_data in model_fits.items():
                    param_str = ", ".join(
                        f"{name} = {val}" for name, val in zip(model_data["param_names"], model_data["params"])
                    )
                    line = f"{model_name}: {param_str} | R² = {model_data['r2']}"
                    fit_lines.append(line)
                    st.success(f"**{line}**")

                # TXT Export for model fits
                fits_txt = "\n".join(fit_lines)
                st.download_button(
                    label="📥 Download Model Fit Summary",
                    data=fits_txt,
                    file_name=f"{gel_name}_model_fits.txt",
                    mime="text/plain",
                    key=f"download_fits_{gel_name}"
                )
            else:
                st.warning("No model fits found for this gel.")

    # Perform deletions after loop
    for gel_name in gels_to_delete:
        del st.session_state["stored_gel_data"][gel_name]
        st.success(f"✅ Deleted {gel_name}")

    # ------------------------------------------------------
    # 🔍 GEL COMPARISON MODE
    # ------------------------------------------------------
    st.markdown("---")
    st.header("🔍 Gel Comparison Mode")

    available_gels = list(st.session_state["stored_gel_data"].keys())
    selected_gels = st.multiselect("Select gels to compare", available_gels)

    if selected_gels and len(selected_gels) >= 2:
        st.subheader("📊 Overlay: Raw Release Profiles")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for gel in selected_gels:
            df = st.session_state["stored_gel_data"][gel].get("release_df", pd.DataFrame())
            if not df.empty:
                ax.errorbar(
                    df["Time (hr)"],
                    df["Actual Conc (mg)"],
                    yerr=df["Std Error"],
                    fmt='o', capsize=4, markersize=5, linewidth=1.5,
                    label=gel
                )
        ax.set_xlabel("Time (hr)", fontsize=12)
        ax.set_ylabel("Actual Dye Concentration (mg)", fontsize=12)
        ax.set_title("Raw Release Profile Comparison", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

        st.subheader("🧬 Compare Model Fits")
        model_choices = ["First-Order", "Higuchi", "Korsmeyer-Peppas"]
        model_to_plot = st.selectbox("Choose a model to compare", model_choices)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for gel in selected_gels:
            data = st.session_state["stored_gel_data"][gel]
            release_df = data.get("release_df", pd.DataFrame())
            model_fits = data.get("model_fits", {})
            if model_to_plot in model_fits and not release_df.empty:
                model_func = {
                    "First-Order": lambda t, C0, k: C0 * (1 - np.exp(-k * t)),
                    "Higuchi": lambda t, kH: kH * np.sqrt(t),
                    "Korsmeyer-Peppas": lambda t, k, n: k * t ** n
                }[model_to_plot]

                t_data = release_df["Time (hr)"].values
                popt = model_fits[model_to_plot]["params"]
                t_fit = np.linspace(min(t_data), max(t_data), 200)
                c_fit = model_func(t_fit, *popt)

                ax2.plot(
                    t_fit, c_fit,
                    label=f"{gel} (R² = {model_fits[model_to_plot]['r2']:.3f})",
                    linewidth=2
                )

        ax2.set_xlabel("Time (hr)", fontsize=12)
        ax2.set_ylabel("Fitted Dye Concentration (mg)", fontsize=12)
        ax2.set_title(f"{model_to_plot} Model Fit Comparison", fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)

        st.subheader(f"📋 {model_to_plot} Parameters Across Gels")

        param_rows = []
        for gel in selected_gels:
            fit_data = st.session_state["stored_gel_data"][gel]["model_fits"].get(model_to_plot)
            if fit_data:
                row = {"Gel Name": gel}
                for name, val in zip(fit_data["param_names"], fit_data["params"]):
                    row[name] = val
                row["R²"] = fit_data["r2"]
                param_rows.append(row)

        if param_rows:
            param_df = pd.DataFrame(param_rows).set_index("Gel Name")
            st.dataframe(param_df, use_container_width=True)
        else:
            st.info("No parameter data available for the selected gels and model.")

render()

import pickle
import os

st.divider()
st.markdown("## 🛠️ Database Tools")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("📤 Export Entire Database as Pickle"):
        with open("stored_gel_data.pkl", "wb") as f:
            pickle.dump(st.session_state["stored_gel_data"], f)
        st.success("Database exported as `stored_gel_data.pkl`")

with col2:
    if st.button("🗑️ Delete All Saved Data"):
        if os.path.exists("stored_gel_data.pkl"):
            os.remove("stored_gel_data.pkl")
        st.session_state["stored_gel_data"] = {}
        st.warning("All saved data has been deleted. Restart the app to refresh.")