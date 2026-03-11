from __future__ import annotations

from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import STRIKE_MARGIN, OK_MARGIN
from ui_helpers.blocks import fig_to_png  



def render_spotcheck_dashboard(df:pd.DataFrame, rig_filter:str, recent_days: int):
    st.title("💧 Spotcheck")

    # 1) Load the DataFrame (do NOT format dates before filtering)
    df_work = df.copy()

    if df is None or df.empty:
        st.info("No spotcheck records found.")
        st.stop()

    # 2) Apply "recent days" cutoff on a working copy
    df_work = df.copy()
    if recent_days and int(recent_days) > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(recent_days))
        df_work = df_work[df_work["Date"] >= cutoff]

    if df_work.empty:
        st.info("No spotcheck entries match the current date filter.")
        st.stop()

    # 3) Build helper numeric columns (keep datetime dtype intact)
    #    target_total_g = (target µL * repeats) / 1000
    df_work["target_total_g"] = (df_work["I) Target vol (µl)"] * df_work["I) Repeat count"]) / 1000.0
    #    ratio_pct = delivered / target_total * 100
    df_work["ratio_pct"] = (df_work["O) Delivered (g)"] / df_work["target_total_g"]) * 100.0
    #    days_since_cal for plotting
    df_work["days_since_cal"] = (df_work["Date"] - df_work["Last calibration"]).dt.total_seconds() / 86400.0

    # ---------------------- OVERVIEW: All rigs ----------------------
    if rig_filter == "All":
        st.subheader("Overview of all rigs")

        st.session_state.setdefault("sc_last_n", 3)
        last_n = st.number_input(
            "Strikes to count in the last N spotchecks",
            min_value=1, value=int(st.session_state.sc_last_n), step=1, key="sc_last_n"
        )

        now = datetime.now(timezone.utc)
        overview_rows = []
        for rig, g in df_work.groupby("Rig"):
            g_sorted = g.sort_values("Date", ascending=False)
            last_cal = g["Last calibration"].max()
            last_sp  = g["Date"].max()

            days_since_cal = (now - last_cal).days if pd.notna(last_cal) else None
            days_since_sp  = (now - last_sp).days if pd.notna(last_sp) else None

            # Spotchecks since the most recent calibration
            n_since_cal = int((g["Date"] >= last_cal).sum()) if pd.notna(last_cal) else 0

            # Total strikes in the filtered window
            total_strikes = int(g["O) Strike?"].sum())

            # Strikes in last N chronologically
            strikes_lastN = int(g_sorted.head(int(last_n))["O) Strike?"].sum())

            overview_rows.append({
                "Rig": rig,
                "Days since calibration": days_since_cal,
                "Days since spotcheck": days_since_sp,
                "Spotchecks since calibration": n_since_cal,
                "Total strikes": total_strikes,
                f"Strikes in last {int(last_n)}": strikes_lastN,
            })

        ov = pd.DataFrame(overview_rows)

        if ov.empty:
            st.info("No rigs to display.")
            st.stop()

        # Sort for readability
        ov = ov.sort_values(
            by=["Days since spotcheck", "Days since calibration", "Rig"],
            ascending=[False, False, True]
        )
        st.dataframe(ov, width='stretch')

        # Optional: raw filtered table
        with st.expander("Show filtered entries (raw table)", expanded=False):
            df_display = df_work.sort_values("Date", ascending=False).copy()
            df_display["Date"] = df_display["Date"].dt.tz_convert(None).dt.strftime("%Y-%m-%d")
            df_display["Last calibration"] = df_display["Last calibration"].dt.tz_convert(None).dt.strftime("%Y-%m-%d")

            display_cols = [
                "Date", "Rig",
                "I) Valve open time (ms)", "I) Target vol (µl)", "I) Repeat count", "I) Target weight (g)",
                "O) Delivered (g)", "O) Mean Drop (µl)", "O) Error %", "O) OK?", "O) Strike?",
                "Last calibration",
            ]
            if "path" in df_display.columns:
                display_cols.append("path")

            st.dataframe(df_display[display_cols], width='stretch')

        st.stop()

    # ---------------------- DETAIL: single rig ----------------------
    st.subheader(f"Rig: {rig_filter}")

    g = df_work[df_work["Rig"] == rig_filter].copy()
    if g.empty:
        st.info("No spotchecks to show for this rig in the current date window.")
        st.stop()

    # Summary KPIs
    now = datetime.now(timezone.utc)
    last_cal = g["Last calibration"].max()
    last_sp  = g["Date"].max()
    days_since_cal = (now - last_cal).days if pd.notna(last_cal) else None
    days_since_sp  = (now - last_sp).days if pd.notna(last_sp) else None
    n_since_cal    = int((g["Date"] >= last_cal).sum()) if pd.notna(last_cal) else 0
    total_strikes  = int(g["O) Strike?"].sum())
    last3_strikes  = int(g.sort_values("Date", ascending=False).head(3)["O) Strike?"].sum())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Days since calibration", f"{days_since_cal if days_since_cal is not None else '—'}")
    with k2:
        st.metric("Spotchecks since calibration", f"{n_since_cal}")
    with k3:
        st.metric("Strikes (total / last 3)", f"{total_strikes} / {last3_strikes}")

    st.divider()

    # ---------- Plot 1: Delivered/Target total (%) vs days since calibration ----------
    plot1 = plt.figure(figsize=(7.5, 4.0), dpi=120)
    ax = plot1.add_subplot(111)

    # Color by strike status
    g["color"] = g["O) OK?"].map({True: "#0d7200", False: "#FF0000"})
    g.loc[g["O) Strike?"] == True, "color"] = "#ffa600"

    ax.scatter(
        g["days_since_cal"], g["ratio_pct"],
        c=g["color"], s=45, edgecolor="white", linewidth=0.6, alpha=0.95, zorder=3
    )

    # Reference lines: OK ±, Strike ±, and 100%
    ax.axhline(100, color="#0d7200", lw=1.0, ls="-", alpha=0.9, zorder=1)
    for y in [(100 - STRIKE_MARGIN), (100 + STRIKE_MARGIN)]:
        ax.axhline(y, color="#ff0000", lw=1.0, ls=":", alpha=0.8, zorder=1)
    for y in [(100 - OK_MARGIN), (100 + OK_MARGIN)]:
        ax.axhline(y, color="#ffa600", lw=1.0, ls="--", alpha=0.8, zorder=1)

    # Optional: adapt y-limits a bit
    y_min = max(0, min(80, float(g["ratio_pct"].min()) - 5))
    y_max = min(200, max(120, float(g["ratio_pct"].max()) + 5))
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Days since last calibration")
    ax.set_ylabel("Delivered / Target total (%)")
    ax.set_title(f"{rig_filter} — Spotcheck ratio vs days since calibration")
    ax.grid(True, ls="--", alpha=0.3)

    png1 = fig_to_png(plot1, cache_key=f"sc_ratio_days|{rig_filter}|{len(g)}|{last_sp.isoformat() if pd.notna(last_sp) else 'NA'}")
    st.image(png1, width='stretch')

    st.divider()

    # ---------- Plot 2: Error % over date ----------
    plot2 = plt.figure(figsize=(7.5, 3.6), dpi=120)
    ax2 = plot2.add_subplot(111)

    x_dates = g["Date"].dt.tz_convert(None)
    ax2.plot(x_dates, g["O) Error %"], color="#7D3C98", lw=1.8, marker="o", ms=4, alpha=0.9)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Error (%)")
    ax2.set_title(f"{rig_filter} — Error % over time")
    ax2.grid(True, ls="--", alpha=0.3)

    png2 = fig_to_png(plot2, cache_key=f"sc_err_date|{rig_filter}|{len(g)}|{last_sp.isoformat() if pd.notna(last_sp) else 'NA'}")
    st.image(png2, width='stretch')

    # ---------- Table: most recent entries ----------
    st.subheader("Most recent spotchecks")
    latest = g.sort_values("Date", ascending=False).head(10).copy()
    latest["Date"] = latest["Date"].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M")
    latest["Last calibration"] = latest["Last calibration"].dt.tz_convert(None).dt.strftime("%Y-%m-%d")

    display_cols = [
        "Date", "I) Valve open time (ms)", "I) Target vol (µl)", "I) Repeat count",
        "O) Delivered (g)", "O) Mean Drop (µl)", "O) Error %", "O) OK?", "O) Strike?",
        "Last calibration",
    ]
    st.dataframe(latest[display_cols], width='stretch')

    # Optional: quick CSV export of filtered rows for this rig
    with st.expander("Export filtered rows"):
        csv = g.sort_values("Date", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name=f"spotchecks_{rig_filter}.csv", mime="text/csv")