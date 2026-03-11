
# pages/watercal_dashboard.py
from __future__ import annotations
import streamlit as st

from services.filters import apply_filters
from ui_helpers.blocks import render_record_block

def render_watercal_dashboard(wcal_ds, rig_filter, recent_days) -> None:

    st.title("💦 Water Calibrations")

    if not wcal_ds.records:
        st.info("No standalone water calibration files found.")
        st.stop()

    filtered = apply_filters(wcal_ds.records, rig_filter, recent_days)

    st.subheader(f"Found {len(filtered)} calibration file(s)")

    # ---- Render list ----
    for i, rec in enumerate(filtered, start=1):
        render_record_block(i, rec)
