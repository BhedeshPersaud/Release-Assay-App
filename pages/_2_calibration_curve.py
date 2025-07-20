import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from utils.state import ensure_state_keys
from utils.helpers import get_absorbance_matrix
from utils.helpers import generate_well_plate_buttons


# -------- Initialize the keys needed by the calibration curve page ------------
ensure_state_keys([
    "uploaded_file",
    "absorbance_matrix",
    "selected_calibration_curve_wells",
    "calibration_model"
])


def render():
    st.title("Calibration Curve 📈")

    # ------ Upload and parse the data from Excel file ------
    uploaded_file = st.file_uploader(
        "Upload Excel file from plate reader",
        type=["xlsx"],
        key="calibration_curve_uploader"
    )
    # check if file is uploaded, and assign it to the session state key we created
    if uploaded_file:
        absorbance_matrix = get_absorbance_matrix(uploaded_file)
        st.session_state.absorbance_matrix = absorbance_matrix

    # If a new file is not uploaded (e.g. when app is re-run during a session, use
    # the absorbance matrix stored in the session state memory
    else:
        absorbance_matrix = st.session_state.get("absorbance_matrix")

    # If no absorbance matrix is stored, give a warning.
    if absorbance_matrix is None:
        st.warning("Please Upload a Plate Reader Excel file to begin!")
        return

    # -------- Select Wells for Calibration -----------------
    st.subheader("Select the wells to build your calibration curve.")
    st.markdown("Click the wells you would like to select.")

    # use the generate_well_plate_buttons to create the grid and return the newly selected wells
    newly_clicked_wells = generate_well_plate_buttons(
        absorbance_matrix=absorbance_matrix,
        key_prefix="calibration_curve_plate"
    )
    selected_wells = set(st.session_state.selected_calibration_curve_wells or [])
    for well in newly_clicked_wells:
        if well in selected_wells:
            # if an already selected well is clicked again, it is removed from the list of
            # selected wells
            selected_wells.remove(well)
        else:
            # otherwise, add the newly clicked well to the selected wells list
            selected_wells.add(well)
    st.session_state.selected_calibration_curve_wells = sorted(selected_wells)

    # --------- Enter concentrations -------------------------
    # If no wells are selected, exit the loop
    if not selected_wells:
        return

    # For all the selected wells, provide a fillable form where the user can input the
    # corresponding dye concentrations of the standards used
    st.subheader("Enter the corresponding concentrations.")
    with st.form("calibration_curve_form"):

        # Initialize an empty dictionary to collect concentration-absorbance pairs for
        # the standards used.
        calibration_well_concentrations = {}

        # Loop through each selected well that is stored in the session state
        for selected_well in st.session_state.selected_calibration_curve_wells:

            # Assign the selected well to a concentration given by a number input
            # provided by the user
            calibration_well_concentrations[selected_well] = st.number_input(
                f"Enter concentration of standard in {selected_well} in µg/mL",
                min_value=0.0,
                step=0.1,
                key=f"conc_{selected_well}"
            )

        # Submit button for the form
        submit_form = st.form_submit_button("Generate Calibration Curve 📈")

    # Exit the loop if form is not submitted yet
    if not submit_form:
        return

    # ------- Build DataFrame for calibration curve -----------

    # Build a list of dictionaries that contain the concentration-absorbance pairs
    calibration_curve_data = []

    # Loop through the calibration_concentrations dictionary and extract the concentrations
    # and absorbances as separate lists
    for selected_well, selected_well_calibration_conc in calibration_well_concentrations.items():

        # Obtain row and column place of well from the well name
        well_row, well_column = selected_well[0], int(selected_well[1:])

        # Use the row and column information to obtain the absorbance value from the
        # absorbance dataframe
        selected_well_absorbance = absorbance_matrix.at[well_row, well_column]

        # Skip wells without any valid absorbance
        if pd.isna(selected_well_absorbance):
            continue

        # For selected wells, add the concentration and absorbance key-value pairs
        # to the calibration_curve_data list
        calibration_curve_data.append({"Concentration": selected_well_calibration_conc,
                                      "Absorbance": float(selected_well_absorbance)
                                       })

    # Build calibration dataframe from the list of dictionaries
    calibration_dataframe = pd.DataFrame(calibration_curve_data)

    # Check that more than 2 data points are selected for calibration i.e. the dataframe
    # contains more than two rows
    if calibration_dataframe.shape[0] < 2:
        st.error("Please select at least two wells to build the calibration curve.")
        return


    # --------------------- Fit Model -------------------------
    # Need a 2D array of x_values, and a 1D array of y values
    x_values = calibration_dataframe[["Concentration"]].values
    y_values = calibration_dataframe["Absorbance"].values

    # Create the calibration model using linear regression
    calibration_model = LinearRegression().fit(x_values, y_values)

    # Save the model to the session state for use in future release profile calculations
    st.session_state.calibration_model = calibration_model

    # Obtain model parameters
    slope, intercept = calibration_model.coef_[0], calibration_model.intercept_
    r2 = calibration_model.score(x_values, y_values)

    # ------------------------ Plot ---------------------------
    # Create a subplot on the page - space to plot the calibration curve
    figure_holder, graph_axes = plt.subplots(figsize=(6,4), dpi=100)

    # Generate the scatter plot using the raw data points
    graph_axes.scatter(calibration_dataframe["Concentration"], calibration_dataframe["Absorbance"], label="Raw data")

    # Overlay the fitted regression model
    graph_axes.plot(
        calibration_dataframe["Concentration"],
        calibration_model.predict(x_values),
        color="green",
        label=f"Fit (R²={r2:.3f})"
    )

    # Title, axis labels and legend
    graph_axes.set_xlabel("Concentration (µg/mL)")
    graph_axes.set_ylabel("Absorbance")
    graph_axes.set_title("Brilliant Blue Calibration Curve")
    graph_axes.legend()
    st.pyplot(figure_holder, use_container_width=False)

    # ------------------ Show equation ------------------------
    st.success(f"Absorbance = {slope:.4f} x Conc.+ {intercept:.4f}")

    return

render()

