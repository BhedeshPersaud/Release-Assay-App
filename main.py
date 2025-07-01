import streamlit as st

st.set_page_config(page_title="Dye Release Analyzer", layout="wide")

st.title("🧪 Welcome to the Dye Release Assay Analyzer")

st.markdown("""
Welcome to your personalized data analysis tool for dye release kinetics and biosensor calibration!

**Capabilities of this app:**
- 📥 Upload and extract absorbance data from a 96-well plate
- 📈 Create a calibration curve with regression analysis
- 🔬 Generate and analyze dye release profiles from PEG hydrogels
- 🧠 Automatically calculate averages, standard errors, and concentrations
- 💾 Store and organize your results in a searchable in-app database

Use the sidebar to navigate between modules:
- **Calibration Curve**
- **Release Profile**
- **Database (Coming Soon)**

Let's get started by uploading your data on the relevant page!
""")
