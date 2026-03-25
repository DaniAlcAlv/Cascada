# cascada.py
# To run> streamlit run src/cascada.py --server.port 8501 --server.address 0.0.0.0
from __future__ import annotations

import streamlit as st

from config import *
from ui.sidebar import render_filters
from ui.page_config import set_page_config
from services.dataset_loader import load_datasets, reload_datasets

from subpages.rig_dashboard import render_rig_dashboard
from subpages.watercal_dashboard import render_watercal_dashboard
from subpages.spotcheck_dashboard import render_spotcheck_dashboard
from subpages.new_spotcheck import render_new_spotcheck
from subpages.manual_calibration import render_manual_calibration
from subpages.launcher import show_launcher

# ----------- Page Config  -----------
set_page_config("CaSCaDa")


# ===================== Sidebar Controls & Page Navigation =====================
# Page selector
page_options = ["🚀 Launcher", "🛠️ Rig dashboard", "💦 Water Calibration dashboard", "💧 Spotcheck dashboard", "➕ New Spotcheck", "✍️ Manual Calibration"]
page = st.sidebar.radio("Module", options=page_options)

# Input directories
with st.sidebar.expander("🗂️ Directories", expanded=False):
    rigs_path      = st.text_input(" 🛠️🗂️ Rig schema folder:", str(RIG_DIR_DEFAULT))
    watercal_path  = st.text_input(" 💦🗂️ Water Calibrations folder:", str(WATERCAL_DIR_DEFAULT))
    spotcheck_path = st.text_input(" 💧🗂️ Spotcheck folder:", str(SPOTCHECK_DIR_DEFAULT))


# ===================== Load Datasets (CACHED) =====================
with st.spinner("⏳ Loading datasets..."):
    rig_ds, wcal_ds, sc_df = load_datasets(rigs_path, watercal_path, spotcheck_path)

# ===================== Sidebar: Rig name and Recency filters =====================
all_rigs = sorted({r.rig_name for r in (rig_ds.records + wcal_ds.records) if r.rig_name})
rig_filter, recent_days = render_filters(all_rigs)

# ===================== Sidebar: Reload datasets button =====================
reload_col = st.sidebar.container()
with reload_col:
    if st.button("🔄 Reload datasets", help="Clear cached datasets and plots, then reload"):
        with st.spinner("Reloading datasets..."):
            reload_datasets()

# ===================== Page: Launcher =====================
if page == "🚀 Launcher":
    show_launcher(rig_ds, wcal_ds, sc_df, spotcheck_path, page_options)

# ===================== Other Pages  =====================
if page == "🛠️ Rig dashboard":
    render_rig_dashboard(rig_ds, wcal_ds,rig_filter, recent_days)

elif page == "💦 Water Calibration dashboard":
    render_watercal_dashboard(wcal_ds, rig_filter, recent_days)

elif page == "💧 Spotcheck dashboard":
    render_spotcheck_dashboard(sc_df, rig_filter, recent_days)

elif page == "➕ New Spotcheck":
    render_new_spotcheck(rig_ds, spotcheck_path, rig_filter)

elif page == "✍️ Manual Calibration":
    render_manual_calibration(rig_ds, rig_filter)