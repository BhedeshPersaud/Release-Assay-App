import pandas as pd
import streamlit as st


# ------Constants / Default Values --------------

# Covers rows 26 to 33, and columns C to N
default_row_range = (25, 34)
default_col_range = (2, 14)

# ------Load the Data ----------------------------

# store the data in a cache to avoid re-reading the same file on
# each run


@st.cache_data
def read_absorbance_matrix(
    uploaded_file,
    row_range=default_row_range,
    col_range=default_col_range
):
    """
    param uploaded_file:
            The excel absorbance file from the plate reader
    param row_range:
            [start, end) indices for the rows of absorbance data
            we are interested in extracting
    param col_range:
            [start, end) indices for the columns of absorbance data
            we are interested in extracting
    return:
        absorbance_matrix:
            an 8x12 DataFrame containing the absorbance values for
            each well of the 96-well plate
    Raises:
        A ValueError if the extracted DataFrame is not 8x12
    """
    # Convert the entire uploaded file into a DataFrame
    plate_reader_df = pd.read_excel(uploaded_file, engine="openpyxl")

    # From the DataFrame above, extract the section of data containing all
    # the absorbance values for the wells
    absorbance_matrix = plate_reader_df.iloc[row_range[0]:row_range[1], col_range[0]:col_range[1]].copy()

    # Check whether the extracted absorbance matrix is 8x12
    #if absorbance_matrix.shape != (8, 12):
        # raise ValueError(f"Expected 8x12 data block, but got {absorbance_matrix.shape}")

    # Label the rows of the extracted absorbance_matrix as A-H, and label the columns as 1-12
    absorbance_matrix.index = [chr(ord("A") + i) for i in range(8)]
    absorbance_matrix.columns = list(range(1, 13))

    return absorbance_matrix


def get_absorbance_matrix(uploaded_file):
    """
    This function is a wrapper around the function read_absorbance_matrix, and includes
    UI error handling.

    param uploaded_file: Excel absorbance file
    return:
        DataFrame of Absorbance matrix, or None if laoding fails, after displaying an
        Error Message
    """

    if uploaded_file is None:
        return None
    try:
        return read_absorbance_matrix(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

# --------------- UI Generation for well buttons  --------------------------------


def generate_well_plate_buttons(
        absorbance_matrix: pd.DataFrame,
        key_prefix: str,
        empty_label: str = "",
        fmt: str = "{:.2f}"
        ):
    """

    param
        absorbance_matrix: the 8x12 absorbance DataFrame with rows A-H and
            columns 1-12.

    param
        key_prefix: Unique prefix for Streamlit widget keys

    param
        empty_label: What to display in the case of an empty well.

    param
        fmt: format string to 2 decimal places for numeric values

    return:
        selected_wells: a list of all the well ID's that were selected
            by the user.
    """
    # Display an error message and return an empty list if there
    # is no absorbance matrix
    if absorbance_matrix is None:
        st.error("No absorbance matrix loaded!")
        return []

    # Initialize selected_wells as an empty list, and determine the number
    # of columns needed
    selected_wells = []
    num_of_cols = len(absorbance_matrix.columns)

    # Create and display a grid of the 96-well plated, where the absorbance values are
    # the labels of the wells
    for row_label in absorbance_matrix.index:
        row_of_cols = st.columns(num_of_cols)
        for col_index, col_label in enumerate(absorbance_matrix.columns):
            well_id = f"{row_label}{col_label}"
            well_value = absorbance_matrix.at[row_label, col_label]

            # Create the button labels for the wells
            # If the well is empty
            if pd.isna(well_value):
                well_button_label = empty_label
            else:
                try:
                    well_button_label = fmt.format(float(well_value))
                except (ValueError, TypeError):
                    well_button_label = str(well_value)

            # Append the well ID to the selected_wells list if that well
            # is clicked by the user.
            button_key = f"{key_prefix}_{well_id}"
            if row_of_cols[col_index].button(well_button_label, key=button_key):
                selected_wells.append(well_id)

    return selected_wells
