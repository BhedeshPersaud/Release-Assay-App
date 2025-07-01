import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.helpers import get_absorbance_matrix
from utils.state import ensure_state_keys

from utils.state import ensure_state_keys
ensure_state_keys(["stored_gel_data"])

if "stored_gel_data" not in st.session_state or st.session_state["stored_gel_data"] is None:
    st.session_state["stored_gel_data"] = {}


st.set_page_config(
    page_title="Calibration Curve",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def render():

    st.title("📈 Release Profile Generator")

    model = st.session_state.get("calibration_model", None)
    absorbance_matrix = st.session_state.get("absorbance_matrix", None)

    if model is None or absorbance_matrix is None:
        st.warning("Please upload a calibration curve and Excel file first.")
        return

    time_axis = st.radio("🕒 What direction represents time points on the plate?", ["Rows", "Columns"], key="time_axis")

    gel_name = st.text_input("Gel Name:", key="gel_name")
    num_trials = st.number_input("How many replicate trials?", min_value=1, max_value=12, value=3, step=1, key="num_trials")
    num_timepoints = st.number_input("How many timepoints per trial?", min_value=1, max_value=8, value=3, step=1, key="num_timepoints")

    time_unit = {"30 min": 0.5, "1 hour": 1, "2 hours": 2}
    time_interval = st.selectbox("Time interval between measurements", list(time_unit.keys()), key="interval_choice")
    interval = time_unit[time_interval]

    st.subheader("🔬 Enter Dilution Factors")
    dilution_factors = []
    for i in range(num_timepoints):
        factor = st.number_input(f"Dilution factor for Timepoint {i + 1}", min_value=0.0, step=0.1, key=f"dilution_{i}")
        dilution_factors.append(factor)

    if "release_well_selections" not in st.session_state:
        st.session_state["release_well_selections"] = {}

    trial_selector = st.selectbox(
        "Currently selecting wells for:",
        range(1, num_trials + 1),
        format_func=lambda x: f"Trial {x}"
    )

    st.subheader("🧪 Select Wells (Shared Plate for All Trials)")

    trial_colors = [cm.get_cmap("Set3")(i) for i in range(num_trials)]
    hex_colors = [
        f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.6)'
        for r, g, b, _ in trial_colors
    ]

    for row_label in absorbance_matrix.index:
        cols = st.columns([1.4] * len(absorbance_matrix.columns))
        for idx, col_label in enumerate(absorbance_matrix.columns):
            well_name = f"{row_label}{col_label}"
            value = absorbance_matrix.loc[row_label, col_label]
            key = f"{gel_name}_Trial{trial_selector}_{well_name}"

            if pd.isna(value):
                cols[idx].button(" ", key=key, disabled=True)
            else:
                try:
                    label = f"{float(value):.3f}"
                except:
                    label = str(value)

                with cols[idx]:
                    # Determine which trial (if any) this well belongs to
                    matching_key = None
                    for t in range(1, num_trials + 1):
                        test_key = f"{gel_name}_Trial{t}_{well_name}"
                        if test_key in st.session_state["release_well_selections"]:
                            matching_key = test_key
                            break

                    # Assign tint if already selected
                    style = (
                        f"background-color:{hex_colors[t - 1]}; color:black; font-weight:bold; "
                        "width: 100%; height: 3.2em; overflow: hidden;"
                        if matching_key else
                        "width: 100%; height: 3.2em; overflow: hidden;"
                    )

                    st.markdown(
                        f"""
                        <style>
                        div[data-testid="stButton"] > button[key="{key}"] {{
                            {style}
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.button(label, key=key, use_container_width=True):
                        if key in st.session_state["release_well_selections"]:
                            st.session_state["release_well_selections"].pop(key)
                        else:
                            st.session_state["release_well_selections"][key] = (row_label, col_label)

    trial_well_sets = []
    for trial in range(1, num_trials + 1):
        trial_keys = [k for k in st.session_state["release_well_selections"] if k.startswith(f"{gel_name}_Trial{trial}_")]
        trial_wells = [st.session_state["release_well_selections"][k] for k in trial_keys]
        trial_well_sets.append(trial_wells)

        if trial_keys:
            st.markdown(f"**Selected wells for Trial {trial}:** {', '.join([k.split('_')[-1] for k in trial_keys])}")

    if st.button("Clear All Selected Release Wells"):
        st.session_state["release_well_selections"] = {}

    release_df = pd.DataFrame()

    if st.button("✅ Generate Release Profile "):
        all_lengths_equal = all(len(t) == num_timepoints for t in trial_well_sets)
        if not all_lengths_equal:
            st.error("All trials must have the same number of time points.")
        else:
            time_points = [i * interval for i in range(1, num_timepoints + 1)]
            table_rows = []

            for i in range(num_timepoints):
                row = {"Time (hr)": time_points[i]}
                abs_values = []
                for trial_idx, trial in enumerate(trial_well_sets):
                    r, c = trial[i]
                    absorb = absorbance_matrix.loc[r, c]
                    row[f"Trial Abs {trial_idx + 1}"] = absorb
                    abs_values.append(absorb)

                row["Avg Abs"] = np.nanmean(abs_values)
                row["Std Error"] = stats.sem(abs_values, nan_policy='omit')

                if not np.isnan(row["Avg Abs"]):
                    row["Dye Conc (mg)"] = float(model.predict(np.array([[row["Avg Abs"]]]))[0])
                else:
                    row["Dye Conc (mg)"] = np.nan

                row["Dilution Factor"] = dilution_factors[i]
                row["Actual Conc (mg)"] = (
                    row["Dye Conc (mg)"] * dilution_factors[i] if not np.isnan(row["Dye Conc (mg)"]) else np.nan
                )

                table_rows.append(row)

            release_df = pd.DataFrame(table_rows)
            st.subheader(f"📊 Release Profile Table for {gel_name}")
            st.dataframe(release_df, use_container_width=True)

            if "stored_gel_data" not in st.session_state:
                st.session_state["stored_gel_data"] = {}
            st.session_state["stored_gel_data"][gel_name] = release_df

    if not release_df.empty:
        from sklearn.metrics import r2_score
        plt.style.use("seaborn-v0_8-whitegrid")

        st.subheader("📊 Dye Release Profile – Raw Experimental Data")
        fig_base, ax_base = plt.subplots(figsize=(8, 5))
        ax_base.errorbar(
            release_df["Time (hr)"],
            release_df["Actual Conc (mg)"],
            yerr=release_df["Std Error"],
            fmt='o', capsize=5, markersize=6, linewidth=2, marker='o',
            color='black', ecolor='gray', label=gel_name
        )
        ax_base.set_xlabel("Time (hr)", fontsize=12, fontweight='bold')
        ax_base.set_ylabel("Actual Dye Concentration (mg)", fontsize=12, fontweight='bold')
        ax_base.set_title(f"Raw Release Profile: {gel_name}", fontsize=14, fontweight='bold')
        ax_base.grid(True, linestyle='--', alpha=0.6)
        ax_base.legend()
        st.pyplot(fig_base)

        # Define models
        def first_order(t, C0, k): return C0 * (1 - np.exp(-k * t))
        def higuchi(t, kH): return kH * np.sqrt(t)
        def korsmeyer_peppas(t, k, n): return k * t ** n

        model_list = [
            ("First-Order", first_order, ["C0", "k"]),
            ("Higuchi", higuchi, ["kH"]),
            ("Korsmeyer-Peppas", korsmeyer_peppas, ["k", "n"])
        ]

        t_data = release_df["Time (hr)"].values
        c_data = release_df["Actual Conc (mg)"].values
        yerr = release_df["Std Error"].values
        t_fit = np.linspace(min(t_data), max(t_data), 200)
        colors = plt.get_cmap("tab10").colors

        # 🧠 Gracefully initialize and save to database BEFORE model fitting
        if "stored_gel_data" not in st.session_state:
            st.session_state["stored_gel_data"] = {}

        # ✅ Proper initialization
        st.session_state["stored_gel_data"][gel_name] = {
            "release_df": release_df,
            "model_fits": {}
        }

        if gel_name not in st.session_state["stored_gel_data"]:
            st.session_state["stored_gel_data"][gel_name] = {}

        st.session_state["stored_gel_data"][gel_name]["release_df"] = release_df

        if "model_fits" not in st.session_state["stored_gel_data"][gel_name]:
            st.session_state["stored_gel_data"][gel_name]["model_fits"] = {}

        for i, (model_name, model_func, param_names) in enumerate(model_list):
            try:
                popt, _ = curve_fit(model_func, t_data, c_data, maxfev=10000)
                c_fit = model_func(t_fit, *popt)
                c_pred = model_func(t_data, *popt)
                r_squared = r2_score(c_data, c_pred)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.errorbar(
                    t_data, c_data, yerr=yerr,
                    fmt='o', capsize=5, markersize=6, marker='o',
                    color=colors[i], ecolor='gray', label="Experimental"
                )
                ax.plot(
                    t_fit, c_fit,
                    linestyle='--', linewidth=2,
                    color=colors[(i + 1) % 10], label=f"{model_name} Fit"
                )

                ax.set_xlabel("Time (hr)", fontsize=12, fontweight='bold')
                ax.set_ylabel("Actual Dye Concentration (mg)", fontsize=12, fontweight='bold')
                ax.set_title(f"{model_name} Model Fit", fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

                st.pyplot(fig)
                param_str = ", ".join(f"{name} = {val:.3f}" for name, val in zip(param_names, popt))
                st.success(
                    f"**{model_name} Fit**: `{model_func.__name__}(t)` with {param_str} | **R² = {r_squared:.3f}**"
                )

                # ✅ Store model fit results
                st.session_state["stored_gel_data"][gel_name]["model_fits"][model_name] = {
                    "params": [round(val, 4) for val in popt],
                    "param_names": param_names,
                    "r2": round(r_squared, 4)
                }

            except Exception as e:
                st.warning(f"⚠️ {model_name} fit failed: {e}")

            # ✅ Final block: Save updated database to disk safely
            import pickle

            try:
                with open("stored_gel_data.pkl", "wb") as f:
                    pickle.dump(st.session_state["stored_gel_data"], f)
                st.success("✅ Gel data saved to disk.")
            except Exception as e:
                st.warning(f"⚠️ Failed to save gel data: {e}")


render()



