# ui.sidebar.py
from __future__ import annotations

import streamlit as st

from config import DAYS_FILTER_DEFAULT


def render_filters(all_rigs: list) -> tuple[str, int]:
    """ Renders the rig dropdown and recent-days filter. """
    # Render controls (dropdown will be refreshed later in pages)
    rig_filter = st.sidebar.selectbox("📦 Filter by rig:", options=["All"]+all_rigs, key="rig_filter")
    recent_days = st.sidebar.number_input(
        "🗓️ Show only records in last N days (0 = all)",
        min_value=0, step=30, key="recent_days", value=DAYS_FILTER_DEFAULT
    )
    return rig_filter, int(recent_days)
