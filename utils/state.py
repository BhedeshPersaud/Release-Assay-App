import streamlit as st
import os
import pickle

# Initialize the names of files that are persistently stored
stored_gel_data_filename = "stored_gel_data.pkl"


def ensure_state_keys(state_keys):
    """
    This function checks whether the 'keys' given as its argument(s)
    are tied to variables that are persistent in the session state of
    the Streamlit app.

        - for known persistent keys, load from disk, or initialize if data
        missing or corrupted
            supported persistent keys:
            * stored_gel_data

        - for unknown keys, initialize an empty dictionary

    param
        state_keys: a list of strings that are the keys for
        variables that are shared between reruns of the Streamlit
        app.

    return: None

    """
    for key in state_keys:
        if key == 'stored_gel_data':
            load_stored_gel_data()
        else:
            if key not in st.session_state:
                st.session_state[key] = {}
    return


def load_stored_gel_data():
    """
    This function is called when it is necessary to load
    the gel data from the stored database.

    return: None
    """
    # Convert the stored gel data filename to a filepath
    filepath = os.path.join(os.getcwd(), stored_gel_data_filename)

    # If that filepath exists already in the current working directory
    # i.e. this project, try opening it.
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                st.session_state["stored_gel_date"] = pickle.load(f)

        # If there is an error, display an error warning.
        except Exception as e:
            st.warning(f"Stored Gel Data could not be loaded: {e}", icon="⚠️")
            st.session_state["stored_gel_data"] = {}

    #  If this filepath does not exist in the cwd, then
    # create a stored_gel_data key for the session state, and initialize
    # it as an empty dictionary.
    else:
        st.session_state["stored_gel_data"] = {}
    return
