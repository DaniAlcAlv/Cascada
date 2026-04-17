from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

from models.spotcheck_model import (
    SpotCheckInput,
    SpotCheckData,
    compute_output,
    save_spotcheck,
)
from models.watercal_dataset import WaterCalDataset
from services.cache import fig_to_png
from services.cache import record_plot_fingerprint
from config import OK_MARGIN, STRIKE_MARGIN

    
def render_new_spotcheck(rig_ds:WaterCalDataset, spotcheck_path:Path|str, rig_filter:str):
    # --- Step 0: Choose a rig (from sidebar filter) ---
    # Use the rig selected in the sidebar
    sel_rig = rig_filter
    spotcheck_path = Path(spotcheck_path)
    st.title("➕ New Spotcheck")

    col_rig, col_cal, col_plot = st.columns([2, 3, 3], vertical_alignment='center')
    with col_rig:
        st.metric(label="Selected Rig", help = "Use sidebar to select another", value=f"{sel_rig}")
        if sel_rig == "All":
            st.error("Select a specific rig on the sidebar")
            st.stop()

    # Build candidate calibrations for this rig to select one for the spotcheck
    candidates = [r for r in rig_ds.records if r.rig_name == sel_rig]
    if not candidates:
        st.error(f"No schemas found for rig '{sel_rig}'.")
        st.stop()
    elif len(candidates) == 1:
        rec = candidates[0]
        with col_cal:
            st.markdown(f"Only one schema found for rig `{sel_rig}`:") 
            st.markdown(f"**{rec.label()}**")
    else: 
        with col_cal:
            selected_idx = st.selectbox(
                "Schema",
                options=list(range(len(candidates))),
                format_func=lambda i: candidates[i].label(),  
                index=0,
            )
        rec = candidates[selected_idx]

    with col_plot:
        fig, _ = rec.plot(show_slope_band=True, draw=False)
        cache_key = record_plot_fingerprint(rec)
        png = fig_to_png(fig, cache_key=cache_key)    
        st.image(png, width="stretch")

    # Step 0.1: Warnings and errors on the selected calibration file
    if rec.errors or len(rec.warnings) > 4:
        st.error(f"Selected calibration has {len(rec.errors)} error(s) and {len(rec.warnings)} warning(s). Update the calibration before proceeding.")   
    
    # Step 1: target + bands
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        target_ul = st.number_input("Target drop volume (µL)", min_value=0.001, value=5.0, step=1.0, format="%.3f")
    with c2:
        ok_margin_pct = st.number_input("OK margin (±%)", min_value=0.0, value=OK_MARGIN, step=0.5, format="%.1f")
    with c3:
        strike_margin_pct = st.number_input("Strike margin (±%)", min_value=0.0, value=STRIKE_MARGIN, step=0.5, format="%.1f")
        if strike_margin_pct < ok_margin_pct:
            st.warning("Strike margin must be ≥ OK margin.")

    # Step 2: ms & repeat count
    c4, c5 = st.columns([1, 1], border=True)
    with c4:
        try:
            inside_bounds, low_ul, high_ul = rec.check_bounds(target_ul, unit="microliters")
            if not inside_bounds:
                st.error(
                    f"Target {target_ul:.0f} µL is outside the calibrated range "
                    f"({low_ul:.0f}–{high_ul:.0f} µL)."
                )
            rec_ms = rec.calc_milliseconds_from_microliters(target_ul)
            st.metric(label="Valve open time (ms)", value=f"{rec_ms:.0f}")
        except Exception as e:
            rec_ms = 0.0
            st.error(f"Could not compute open time: {e}")

    with c5:
        repeat_count = abs(1000.0 / target_ul)
        repeat_count = max(1, repeat_count)
        st.metric(label="Repeat count (for ≈1 g total)", value=repeat_count)

    st.divider()

    # Step 3: measured total (g)
    measured_g = st.number_input("Measured total (g)", min_value=0.0, value=0.0, step=0.01)

    # Step 4: compute output using your function
    output = None
    if measured_g > 0.0 and target_ul > 0.0 and strike_margin_pct >= ok_margin_pct:
        sc_in = SpotCheckInput(
            valve_open_time_ms=float(rec_ms),
            target_volume_microliters=float(target_ul),
            repeat_count=int(repeat_count),
            ok_margin_pct=float(ok_margin_pct),
            strike_margin_pct=float(strike_margin_pct),
        )
        try:
            output = compute_output(sc_in, total_delivered_grams=float(measured_g))
            status = "OK" if output.ok else ("Strike" if output.strike else "Failed")
            emoji  = "✅" if output.ok else ("⚠️" if output.strike else "❌")

            c6, c7, c8 = st.columns([1, 1, 1], border=True)
            with c6:
                st.metric("Mean drop (µL)", f"{output.drop_volume_microliters:.1f}")
            with c7:
                st.metric("Error (%)", f"{output.error_percentage:.2f}%")
            with c8:
                st.markdown(f"# {emoji} **{status}**")
        except Exception as e:
            st.error(f"Failed to compute spotcheck output: {e}")
    else:
        st.info("Enter a target volume (>0) and a measured total (>0) to compute results.")

    # Step 5: notes & save
    notes_txt = st.text_area("Notes (optional)", value="", placeholder="User name, observations, etc.")
    can_save = (output is not None)

    save_disabled_reason = None
    if not can_save:
        save_disabled_reason = "Enter a valid target, compute recommended values, and input the measured total."
    if strike_margin_pct < ok_margin_pct:
        save_disabled_reason = "Strike margin must be ≥ OK margin."

    save_col, _ = st.columns([1, 3])
    with save_col:
        if st.button("💾 Save spotcheck", disabled=bool(save_disabled_reason)):
            try:
                now_utc = datetime.now(timezone.utc)
                last_cal_dt = rec.date
                if last_cal_dt is not None and last_cal_dt.tzinfo is None:
                    last_cal_dt = last_cal_dt.replace(tzinfo=timezone.utc)

                sc = SpotCheckData(
                    date=now_utc,
                    rig_name=sel_rig,
                    last_calibration_date=last_cal_dt or now_utc,
                    notes=notes_txt or None,
                    input=sc_in,           # from Step 4
                    output=output,         # from Step 4
                )
                target_path = save_spotcheck(spotcheck_path, sc)
                st.success(f"Spotcheck saved: `{target_path}`")
            except Exception as e:
                st.error(f"Failed to save spotcheck: {e}")