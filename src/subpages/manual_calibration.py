from datetime import datetime, timezone
from typing import List
import streamlit as st

from models.watercal_model import (
    WaterCalRecord,
    WaterCalInput,
    WaterCalMeasure,
    WaterValveCalibration,
)
from models.watercal_dataset import WaterCalDataset

from services.cache import fig_to_png


def render_manual_calibration(rig_ds:WaterCalDataset, rig_filter:str):
    st.set_page_config(
        page_title="✍️ Manual Calibration",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("✍️ Manual Calibration")

    # Require a specific rig from the sidebar filter
    if rig_filter == "All":
        st.error("Please select a specific rig in the sidebar to create a manual calibration.")
        st.stop()

    # Remember computed calibration across reruns
    st.session_state.setdefault("mc_calibration", None)

    # (Rig + computer name)
    col_rig, col_comp = st.columns([1, 1], border=True)
    with col_rig:
        sel_rig = rig_filter
        st.markdown(f"**Rig (from sidebar):** `{sel_rig}`")
        st.caption("Change the rig using the sidebar 'Filter by rig'.")

    rig_candidates = [r for r in rig_ds.records if r.rig_name == sel_rig]
    inferred_computer = rig_candidates[0].computer_name if rig_candidates else "(unknown-computer)"
    with col_comp:
        computer_name = st.text_input("Computer name", value=inferred_computer)

    help_str = (
        "Add at least **two** rows. Each row needs:\n"
        "- **Valve open interval (s)** (e.g., 0.5)\n"
        "- **Valve open time (s)** (e.g., 0.025)\n"
        "- **Sample weight (g)** accepts a single value or a series as a comma-separated list (e.g., `0.39, 0.40, 0.38`)\n"
        "- **Repeat count** (n ≥ 21)"
    )
    st.subheader("1) Enter measurement rows", help=help_str)

    # Keep rows in session state
    if "mc_rows" not in st.session_state:
        st.session_state.mc_rows = [
            {"interval_s": 0.5, "open_time_s": 0.025, "weight": "", "repeat_count": 200},
            {"interval_s": 0.5, "open_time_s": 0.050, "weight": "", "repeat_count": 200},
        ]

    def render_mc_row(idx: int, row: dict):
        c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
        with c1:
            st.session_state.mc_rows[idx]["interval_s"] = st.number_input(
                "Valve open interval (s)",
                min_value=0.000001,
                value=float(row["interval_s"]),
                step=0.01,
                key=f"mc_int_{idx}",
                format="%.6f",
            )
        with c2:
            st.session_state.mc_rows[idx]["open_time_s"] = st.number_input(
                "Valve open time (s)",
                min_value=0.000001,
                value=float(row["open_time_s"]),
                step=0.001,
                key=f"mc_open_{idx}",
                format="%.6f",
            )
        with c3:
            st.session_state.mc_rows[idx]["weight"] = st.text_input(
                "Sample weight (g), comma-separated",
                value=row["weight"],
                key=f"mc_w_{idx}",
            )
        with c4:
            st.session_state.mc_rows[idx]["repeat_count"] = st.number_input(
                "Repeat count (≥ 21)",
                min_value=21,
                value=int(row["repeat_count"]),
                step=1,
                key=f"mc_rep_{idx}",
            )

    # Edit rows
    for i, r in enumerate(st.session_state.mc_rows):
        with st.container(border=True):
            st.markdown(f"**Row {i+1}**")
            render_mc_row(i, r)
            rm_col, _ = st.columns([1, 6])
            with rm_col:
                if st.button("Remove row", key=f"rm_{i}"):
                    st.session_state.mc_rows.pop(i)
                    # force redraw now so keys/length are consistent
                    st.rerun()

    # Add row (ensure numeric default; rerun immediately)
    add_col, _ = st.columns([1, 6])
    with add_col:
        if st.button("➕ Add row"):
            next_ot = (
                float(st.session_state.mc_rows[-1]["open_time_s"]) + 0.025
                if st.session_state.mc_rows else 0.025
            )
            st.session_state.mc_rows.append(
                {"interval_s": 0.5, "open_time_s": round(next_ot, 6), "weight": "", "repeat_count": 200}
            )
            st.rerun()

    st.divider()

    # Build calibration from input
    st.subheader("2) Compute calibration from input")
    description = st.text_input("Description (optional)", value="Manual calibration entry")
    notes = st.text_area("Notes (optional)", value="", placeholder="Technician, lab conditions, etc.")

    def _parse_weights(txt: str) -> List[float]:
        parts = [w.strip() for w in txt.split(",")]
        return [float(p) for p in parts if p]

    if st.button("Compute calibration"):
        try:
            measures: List[WaterCalMeasure] = []
            for row in st.session_state.mc_rows:
                weights = _parse_weights(row["weight"])
                if len(weights) == 0:
                    raise ValueError("Each row must have at least one weight value.")
                measures.append(
                    WaterCalMeasure(
                        valve_open_interval=float(row["interval_s"]),
                        valve_open_time=float(row["open_time_s"]),
                        water_weight=weights,
                        repeat_count=int(row["repeat_count"]),
                    )
                )

            cal_input = WaterCalInput(measurements=measures)
            calibration = WaterValveCalibration.from_input(
                cal_input=cal_input,
                date=datetime.now(timezone.utc),
                description=description or None,
                notes=notes or None,
            )
            # Persist for the save step (survives next rerun)
            st.session_state.mc_calibration = calibration

            # Optional: scroll to results; simply rerun is enough to render below
            st.rerun()

        except Exception as e:
            st.error(f"Failed to compute calibration: {e}")

    # Results & Save are driven by the persisted object (not by the compute button state)
    calibration = st.session_state.mc_calibration
    if calibration is not None:
        s, o, r2 = calibration.preferred_coefficients
        c1, c2, c3 = st.columns([1, 1, 1], border=True)
        with c1: st.metric("Slope", f"{s:.6f}")
        with c2: st.metric("Offset", f"{o:+.6f}")
        with c3: st.metric("R²", f"{r2:.4f}")

        if calibration.errors:
            st.error("Errors:\n\n- " + "\n- ".join(calibration.errors))
        if calibration.warnings:
            st.warning("Warnings:\n\n- " + "\n- ".join(calibration.warnings))

        try:
            fig, _ = calibration.plot(draw=False, title=f"{sel_rig} — Manual Calibration")
            cache_key = f"manual|{s:.6f}|{o:.6f}|{r2:.6f}"
            png = fig_to_png(fig, cache_key=cache_key)
            st.image(png, width="stretch")
        except Exception as e:
            st.warning(f"Plot not available: {e}")

        st.divider()

        # Save using WaterCalRecord.save_manual_calibration(...)
        st.subheader("3) Save calibration (folder with water_calibration.json + rig_info.json)")
        save_root = st.text_input("Save root folder", value=st.session_state.get("wcal_dir", "WATERCAL_DIR_DEFAULT"))
        custom_record_id = st.text_input("Folder name (optional)", value=f"{sel_rig}_manual")

        col_save, col_actions = st.columns([1, 2])
        with col_save:
            if st.button("💾 Save calibration folder"):
                try:
                    target_dir = WaterCalRecord.save_manual_calibration(
                        save_root,
                        computer_name=computer_name,
                        rig_name=sel_rig,
                        calibration=calibration,
                        record_id=(custom_record_id or None),
                    )
                    st.success(f"Calibration saved under: `{target_dir}`")
                except Exception as e:
                    st.error(f"Failed to save calibration: {e}")

        # Optional actions: clear / reload
        with col_actions:
            if st.button("Clear computed calibration"):
                st.session_state.mc_calibration = None
                st.rerun()