# CaSCADa.py
# To run> streamlit run src/cascada.py --server.port 8501 --server.address 0.0.0.0

from __future__ import annotations
import logging
import streamlit as st
import pandas as pd #Only for Typehints

from services.filesystem import find_repo_root
from models.watercal_dataset import WaterCalDataset
from ui_helpers.blocks import show_skipped_files

def show_launcher(rig_ds: WaterCalDataset, wcal_ds: WaterCalDataset, spotcheck_df: pd.DataFrame, spotcheck_path:str, page_options: list) -> None:
    st.set_page_config(
        page_title="💧 CaSCaDa Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # ---- Main title ----
    st.title("CaSCaDa")
    st.write("Calibration and Spot Check assistance Dashboard")
    st.divider()

    # ---- Access to other pages ----
    st.write("Choose one of the following modules on the sidebar")
    col1, col2, col3 = st.columns(3, border=True) 
    with col1:
        st.markdown("#### 🛠️ Rig Schemas")
        st.caption("View and update the calibration data stored in the rig schema.")

    with col2:
        st.markdown("#### 💦 Calibrations")
        st.caption("Browse all calibration files, check regression quality.")

    with col3:
        st.markdown("#### 💧 Spotchecks")
        st.caption("Browse spot checks.")


    col11, col12 = st.columns(2, border=True)  
    with col11:
        st.markdown("#### ✍️ Enter Manual Calibration")
        st.caption("Enter measurements to create and store a new calibration.")

    with col12:
        st.markdown("#### ➕ Enter new Spotcheck")
        st.caption("Compute OK/Strike/Fail and save a spotcheck entry.")


    # ---- Status summary ----
    st.markdown("#### 📊 Data Successfully Loaded")
    path_cols = st.columns(3)
    with path_cols[0]:
        st.metric("Rig schemas", f"{len(rig_ds.records)} files")
        st.caption(f"`{rig_ds.main_dir}`")

    with path_cols[1]:
        st.metric("WaterCal calibrations", f"{len(wcal_ds.records)} files")
        st.caption(f"`{wcal_ds.main_dir}`")

    with path_cols[2]:
        st.metric("Spotcheck records", f"{len(spotcheck_df)} files")
        st.caption(f"`{spotcheck_path}`")

    # ---- Skipped files block ----
    show_skipped_files(rig_ds, wcal_ds)

    # ---- About / version block ----
    with st.expander("ℹ️ About CaSCaDa", expanded=False):
        repo_root = find_repo_root()
        if not repo_root:
            st.error("Repository root not found.")
            st.stop()

        readme_path = repo_root / "README.md"
        try:
            md = readme_path.read_text(encoding="utf-8")
            st.markdown(md)
        except Exception as e:
            st.error(f"Error reading README.md: {e}")
            st.stop()