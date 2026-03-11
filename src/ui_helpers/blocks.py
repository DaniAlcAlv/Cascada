# blocks.py

import streamlit as st

from services.cache import record_plot_fingerprint, fig_to_png
from models.watercal_model import WaterCalRecord

from models.watercal_dataset import WaterCalDataset 

def render_record_block(i: int, rec: WaterCalRecord):
    """Shared detail + plot block for a single record."""

    with st.expander(rec.label(), expanded=False):
        cols = st.columns([2, 3])

        with cols[0]:
            # Path
            st.write(f"Path: *`{rec.file_path}`*")
            
            # Warnings/Errors
            if rec.warnings:
                st.warning("Warnings:\n- " + "\n- ".join(rec.warnings))
            if rec.errors:
                st.error("Errors:\n- " + "\n- ".join(rec.errors))
            if not rec.warnings and not rec.errors:
                st.success("No issues found in this record.")

            # Calculator
            microliters = st.number_input(
                "Volume (µL) → Valve open time (ms):",
                min_value=0.0,
                step=1.0,
                key=f"vol_{i}"
            )
            if microliters > 0:
                inside_bounds, low_vol, hi_vol = rec.check_bounds(microliters, unit="microliters")
                if not inside_bounds:
                    st.warning(f"{microliters:.0f} µL is outside calibrated range ({low_vol:.0f}–{hi_vol:.0f} µL).")
                else:
                    t_ms = rec.calc_milliseconds_from_microliters(microliters)
                    st.success(f"Estimated valve open time: **{t_ms:.2f} ms**")

        with cols[1]:
            show_band = True
            fig, _ = rec.plot(show_slope_band=show_band, draw=False)
            cache_key = record_plot_fingerprint(rec)
            png = fig_to_png(fig, cache_key=cache_key)     # fig ignored for hashing; cache_key drives cache
            st.image(png, width="stretch")                 



def show_skipped_files(rig_ds:WaterCalDataset, wcal_ds:WaterCalDataset): 
    """Added a skipped files info (both datasets if available)"""
    with st.expander("Files unable to load", expanded=True):
        if rig_ds.skipped_files:
            with st.sidebar.expander(f"Rigs - Skipped files: {len(rig_ds.skipped_files)}", expanded=False):
                for p, reason in rig_ds.skipped_files[:20]:
                    st.write(f"- `{p}` — {reason}")
                if len(rig_ds.skipped_files) > 20:
                    st.caption(f"... and {len(rig_ds.skipped_files) - 20} more")

        if wcal_ds.skipped_files:
            with st.sidebar.expander(f"WaterCal - Skipped files: {len(wcal_ds.skipped_files)}", expanded=False):
                for p, reason in wcal_ds.skipped_files[:20]:
                    st.write(f"- `{p}` — {reason}")
                if len(wcal_ds.skipped_files) > 20:
                    st.caption(f"... and {len(wcal_ds.skipped_files) - 20} more")