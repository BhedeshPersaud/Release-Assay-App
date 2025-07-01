import streamlit as st
import pandas as pd

def get_absorbance_matrix(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
        abs_matrix = df.iloc[25:33, 2:14]  # C26 to N33 inclusive
        abs_matrix.index = list("ABCDEFGH")  # 8 rows
        abs_matrix.columns = list(range(1, 13))  # 12 columns
        return abs_matrix
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def generate_96_well_plate(abs_matrix, key_prefix):
    if abs_matrix is None:
        st.warning("No absorbance matrix loaded.")
        return []

    selected_wells = []

    cols_labels = list(abs_matrix.columns)
    row_labels = list(abs_matrix.index)

    for row_idx, row_label in enumerate(row_labels):
        cols = st.columns(len(cols_labels))
        for col_idx, col_label in enumerate(cols_labels):
            well = f"{row_label}{col_label}"
            value = abs_matrix.loc[row_label, col_label]
            try:
                display = "" if pd.isna(value) else f"{float(value):.2f}"
            except (ValueError, TypeError):
                display = str(value)
            button_label = display
            button_key = f"{key_prefix}_{well}"
            if cols[col_idx].button(button_label, key=button_key):
                selected_wells.append(well)

    return selected_wells
