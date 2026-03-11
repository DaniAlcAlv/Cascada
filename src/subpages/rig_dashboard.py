from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

from models.watercal_model import WaterCalRecord
from models.watercal_dataset import WaterCalDataset

from ui_helpers.blocks import render_record_block
from services.filters import apply_filters
from services.dataset_loader import reload_datasets


def render_rig_dashboard(rig_ds:WaterCalDataset, wcal_ds: WaterCalDataset, rig_filter:str, recent_days:int):
    st.title("🛠️ WaterCal in rig schema")
    if not rig_ds.records:
        st.info("Rig schemas not found.")
        st.stop()

    # ---------- Existing list rendering ----------
    filtered = apply_filters(rig_ds.records, rig_filter, recent_days)
    st.subheader(f"Found {len(filtered)} calibration file(s)")
    for i, rec in enumerate(filtered, 1):
        render_record_block(i, rec)

    # ---------- Update rig schema from a WaterCal file ----------
    st.subheader("Update rig schema from a WaterCal file")

    if rig_filter == "All":
        st.warning("Select a specific rig in the sidebar to enable updates.")
    else:
        sel_rig = rig_filter

        # 1) Choose target rig schema entry (from rig_ds for this rig)
        rig_targets = [r for r in rig_ds.records if r.rig_name == sel_rig]
        if not rig_targets:
            st.info(f"No rig schema files for rig '{sel_rig}'.")
        else:
            tgt_idx = st.selectbox(
                "Target rig schema file",
                options=list(range(len(rig_targets))),
                format_func=lambda i: rig_targets[i].label(),
                index=0,
            )
            tgt_rec = rig_targets[tgt_idx]

            # We will write directly to this path (either an envelope JSON or water_calibration.json)
            tgt_path: Optional[Path] = getattr(tgt_rec, "file_path", None)
            if tgt_path is None:
                st.error("Selected rig record has no file_path; cannot update.")
            else:
                st.caption(f"Target: `{tgt_path}`")

            # 2) Choose WaterCal source (prefer standalone water_cal, then fallback to rig-based)
            wcal_candidates = [r for r in wcal_ds.records if r.rig_name == sel_rig]
            if not wcal_candidates:
                wcal_candidates = [r for r in rig_ds.records if r.rig_name == sel_rig]

            if not wcal_candidates:
                st.info(f"No WaterCal files found for rig '{sel_rig}'.")
                src_rec = None
            else:
                # default to latest
                def _rec_date(r:WaterCalRecord):
                    return r.date or datetime.min.replace(tzinfo=timezone.utc)

                wcal_candidates = sorted(wcal_candidates, key=_rec_date, reverse=True)

                src_idx = st.selectbox(
                    "WaterCal source file",
                    options=list(range(len(wcal_candidates))),
                    format_func=lambda i: wcal_candidates[i].label(),
                    index=0,
                )
                src_rec = wcal_candidates[src_idx]

            # 3) Apply update using watercal_model only
            apply_col, _ = st.columns([1, 3])
            with apply_col:
                can_apply = (src_rec is not None) and (tgt_path is not None)
                if st.button("📝 Update rig schema", disabled=not can_apply):
                    try:
                        written_path = tgt_rec.update_calibration_json(
                            new_calibration=src_rec.calibration.water_valve,
                            target=tgt_path,          # can be envelope JSON or water_calibration.json
                            make_backup=True
                        )
                        
                        st.success(f"Updated: `{written_path}`")
                        reload_datasets() # Reload rig dataset so UI reflects updated file

                    except Exception as e:
                        st.error(f"Failed to update rig schema: {e}")