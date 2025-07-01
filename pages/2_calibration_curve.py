from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


from utils.helpers import get_absorbance_matrix
from utils.state import ensure_state_keys



# ✅ Set page to wide layout and collapse sidebar
st.set_page_config(
    page_title="Calibration Curve",
    layout="wide",
    initial_sidebar_state="collapsed"
)
def render():

    st.title("🧪 Calibration Curve")

    # 💡 FIX: Add CSS to prevent absorbance label wrapping
    st.markdown("""
        <style>
        button[kind="secondary"] > div > div {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 0.85rem;
            padding: 0.4em 0.7em;
            min-width: 3em;
        }
        </style>
    """, unsafe_allow_html=True)

    ensure_state_keys([
        "uploaded_file",
        "absorbance_matrix",
        "calibration_wells",
        "calibration_model"
    ])

    uploaded_file = st.file_uploader("Upload 96-well plate Excel file", type=["xlsx"])
    if uploaded_file:
        absorbance_matrix = get_absorbance_matrix(uploaded_file)
        st.session_state.absorbance_matrix = absorbance_matrix
        st.session_state.uploaded_file = uploaded_file
    else:
        absorbance_matrix = st.session_state.get("absorbance_matrix", None)

    if absorbance_matrix is None:
        st.warning("Please upload an Excel file to begin.")
        return

    if "calibration_wells" not in st.session_state or st.session_state.calibration_wells is None:
        st.session_state.calibration_wells = []

    st.subheader("2. Select Wells for Calibration")

    if st.button("Reset Selected Wells"):
        st.session_state.calibration_wells = []

    for row_label in absorbance_matrix.index:
        cols = st.columns([1.4] * len(absorbance_matrix.columns))
        for idx, col_label in enumerate(absorbance_matrix.columns):
            well_name = f"{row_label}{col_label}"
            value = absorbance_matrix.loc[row_label, col_label]

            if pd.isna(value):
                cols[idx].button(" ", key=f"plate_{well_name}", disabled=True)
            else:
                try:
                    label = f"{float(value):.3f}"
                    is_valid = True
                except:
                    is_valid = False

                if is_valid:
                    is_selected = well_name in st.session_state.calibration_wells
                    btn_style = (
                        "background-color:#cce5ff; color:black; font-weight:bold;"
                        if is_selected else ""
                    )

                    with cols[idx]:
                        st.markdown(
                            f"""
                            <style>
                            div[data-testid="stButton"] > button[key="plate_{well_name}"] {{
                                {btn_style}
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button(label, key=f"plate_{well_name}", use_container_width=True):
                            if is_selected:
                                st.session_state.calibration_wells.remove(well_name)
                            else:
                                st.session_state.calibration_wells.append(well_name)

    if st.session_state.calibration_wells:
        st.info(f"🧬 Selected wells: {', '.join(st.session_state.calibration_wells)}")
    else:
        st.warning("No wells selected yet.")

    st.subheader("3. Enter Known Concentrations")

    concentrations = []
    absorbances = []

    for well in st.session_state.calibration_wells:
        row = well[0]
        col = well[1:]
        col = int(col) if col.isdigit() else col

        try:
            absorb = float(absorbance_matrix.loc[row, col])
        except:
            absorb = np.nan

        col1, col2 = st.columns([1, 2])
        with col1:
            conc = st.number_input(f"Concentration for {well}", min_value=0.0, step=0.1, key=f"conc_{well}")
        with col2:
            st.markdown(f"**Absorbance:** {absorb:.3f}" if not np.isnan(absorb) else "**Absorbance:** -")

        concentrations.append(conc)
        absorbances.append(absorb)

    if st.button("Generate Calibration Curve"):
        st.subheader("📊 Calibration Curve")
        cal_df = pd.DataFrame({
            "Concentration": concentrations,
            "Absorbance": absorbances
        }).dropna()

        if len(cal_df) < 2:
            st.error("Please select at least 2 valid wells with concentrations and absorbances.")
        else:
            model = LinearRegression()
            x = cal_df["Concentration"].values.reshape(-1, 1)
            y = cal_df["Absorbance"].values
            model.fit(x, y)

            slope = model.coef_[0]
            intercept = model.intercept_
            r2 = model.score(x, y)

            st.session_state["calibration_model"] = model

            fig, ax = plt.subplots()
            ax.scatter(cal_df["Concentration"], cal_df["Absorbance"], label="Data")
            ax.plot(cal_df["Concentration"], model.predict(x), color="red", label="Fit")
            ax.set_xlabel("Concentration (mg)")
            ax.set_ylabel("Absorbance")
            ax.set_title("Calibration Curve")
            ax.legend()
            st.pyplot(fig)

            st.success(f"**Regression Equation:** Absorbance = {slope:.4f} × Conc + {intercept:.4f}")
            st.info(f"**R² Value:** {r2:.4f}")

render()