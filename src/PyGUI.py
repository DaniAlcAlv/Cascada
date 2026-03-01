# PyGUI.py

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional

from watercal_dataset import WaterCalDataset
from watercal_model import WaterCalRecord

import logging
logging.basicConfig(
    level=logging.INFO,       # INFO shows skipped file messages; DEBUG for more detail
    format="%(levelname)s: %(message)s"
)


MENU_WIDTH = 90  # At least 80 for decent layout

# Default directory can be injected by caller; kept here for convenience
DEFAULT_MAIN_DIR = Path(r"C:\Data\Rig jsons 2026-2-19")

# ---------- ANSI helpers ----------
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def visible_len(s: str) -> int:
    """Length without ANSI escape sequences (approx mono-space width)."""
    return len(ANSI_RE.sub("", s))


# ---------- Menu Rendering ----------
def render_main_menu(dataset: WaterCalDataset, menu_width: int = MENU_WIDTH) -> List[Tuple[str, WaterCalRecord]]:
    """
    Prints a grouped (by rig_num) single-level menu and returns a flat list mapping
    (rig_num, record) for selection (1..N).
    """
    title = " WATER CALIBRATIONS FOUND IN RIG JSONS "
    header = f"\n┌{title:─^{menu_width-2}}┐"
    footer = f"└{'':─^{menu_width-2}}┘"

    def row(text: str = "") -> str:
        # Wrapper adds no extra left space: "│" + content + "│"
        return f"│{text.ljust(menu_width-2)}│"

    def section_title(rig_num: str, count: int) -> str:
        label = f"├─Other Rigs───({count} file{'s' if count != 1 else ''})" if rig_num == "Other" \
            else f"├─Rigs {rig_num}───({count} file{'s' if count != 1 else ''})"
        return f"{label:─<{menu_width-1}}┤"

    def entry_line(idx: int, rec: WaterCalRecord) -> str:
        # Colored flags
        warn_raw = f"\033[93m[{rec.n_warnings}!]\033[0m" if rec.warnings else "." * 4
        err_raw  = f"\033[91m[{rec.n_errors}!]\033[0m"   if rec.errors   else "." * 4

        # Center warn+err into 8 visible columns (ANSI-safe)
        flags_width = 8
        flags_visible = visible_len(warn_raw + err_raw)
        pad_left = max(0, (flags_width - flags_visible) // 2)
        pad_right = max(0, flags_width - flags_visible - pad_left)
        flags_col = " " * pad_left + warn_raw + err_raw + " " * pad_right

        # Fixed-width columns
        idx_col = f"{idx:>4})"  # 5 chars

        # Formula (centered in 38 chars)
        slope, offset, r2 = rec.preferred_coefficients
        formula_text = (f"W={slope:0.5f}·t{offset:+0.5f}..(R²={r2:0.4f})")
        formula_col = f"{formula_text:.^38}"

        # Date (11 chars typical 'YYYY-MM-DD' plus space)
        date_str = rec.date.strftime("%Y-%m-%d") if rec.date else "..NoDate.."
        date_col = f"{date_str} "

        # Compute space left for the file column (ANSI-safe for flags)
        occupied = len(idx_col) + visible_len(flags_col) + len(formula_col) + len(date_col)
        available_for_file = (menu_width - 2) - occupied  # subtract borders already in row()
        if available_for_file < 10:
            available_for_file = 10

        id = (rec.record_id[: available_for_file - 1] + "…") if len(rec.record_id) > available_for_file else rec.record_id
        file_col = f"{id:.<{available_for_file}}"

        # Format: Idx + file_name + flags + formula + Date
        return row(f"{idx_col}{file_col}{flags_col}{formula_col}{date_col}")

    # Build groups and render
    groups = dataset.ordered_rig_num_groups()  # List[Tuple[rig_num, List[WaterCalRecord]]]

    print(header)
    if dataset.skipped_files:
        # Show a concise skipped summary under the header
        print(row(f"Note: skipped {len(dataset.skipped_files)} file(s). Use --verbose logging to see why."))

    flat: List[Tuple[str, WaterCalRecord]] = []
    idx = 1
    for rig_num, group in groups:
        print(section_title(rig_num, len(group)))
        if not group:
            print(row("  └─ (no files)"))
        else:
            for rec in group:
                print(entry_line(idx, rec))
                flat.append((rig_num, rec))
                idx += 1

    print(footer)
    print("0 (or empty) to exit")
    return flat


def render_record_details(rec: WaterCalRecord, width: int = MENU_WIDTH) -> str:
    """Pretty record detail block for terminal."""
    title = f" Details — {rec.rig_name} @ {rec.computer_name} "
    header = f"┌{title:─^{width-2}}┐"
    footer = f"└{'':─^{width-2}}┘"

    def row(text: str = "", special_char_mod: int = 0) -> str:
        # This version uses one leading space inside the left border
        return f"│ {text.ljust(width - 3 + special_char_mod)}│"

    # Coefficients
    s0, o0, r20 = rec.cal_output.slope, rec.cal_output.offset, rec.cal_output.r2
    s1, o1, r21 = rec.recomputed.slope, rec.recomputed.offset, rec.recomputed.r2
    date_str = rec.date.strftime("%Y-%m-%d") if rec.date else "NoDate"

    lines = [
        header,
        row(f"File: {rec.rig_name}"),
        row(f"Date: {date_str}"),
        f"├{'─'*(width-2)}┤",
        row(f"Original:   slope {s0:>9.6f}    offset {o0:>9.6f}     R² {r20:>5.3f}"),
        row(f"Recomputed: slope {s1:>9.6f}    offset {o1:>9.6f}     R² {r21:>5.3f}"),
    ]

    if rec.warnings:
        lines.append(f"├{'─'*(width-2)}┤")
        warn_title = (f"\033[93m⚠️  WARNINGS \033[0m")
        lines.append(row(warn_title, special_char_mod=10))
        for w in rec.warnings:
            lines.append(row(f"- {w}"))

    if rec.errors:
        lines.append(f"├{'─'*(width-2)}┤")
        err_title = f"\033[91m❌ ERRORS \033[0m "
        lines.append(row(err_title, special_char_mod=8))
        for e in rec.errors:
            lines.append(row(f"- {e}"))

    lines.append(footer)
    return "\n".join(lines)

# ---------- Calculator Prompt ----------
def prompt_volume_and_calculate(record: WaterCalRecord) -> None:
    while True:
        ul_str = input("Enter a volume in µL to get valve open time (ms) [blank or 0 to go back]: ").strip()
        if not ul_str or ul_str == "0":
            break
        try:
            microliters = float(ul_str)
        except ValueError:
            print("❌ Invalid number.")
            continue

        inside_bounds, low_vol, hi_vol = record.check_bounds(microliters, unit="microliters")
        if inside_bounds:
            ms = record.calc_milliseconds_from_microliters(microliters)
            print(f" => {ms:.2f} ms")
        else:
            print(f" => ❌ {microliters:g} µL is outside the calibrated range {low_vol:.1f}–{hi_vol:.1f} µL")
    print("(Back to list. Press 0 to exit.)")


# ---------- App ----------
def interactive_app(main_dir: Optional[Path] = None):
    # main_dir = main_dir or Path(".")
    # dataset = WaterCalDataset.load_from_rigs(main_dir)
    dataset = WaterCalDataset.load_from_water_cal_dir(r"C:\Data\Water-cal")

    if not dataset.records:
        print("No water calibration information found.")
        if dataset.skipped_files:
            print(f"(Note: skipped {len(dataset.skipped_files)} file(s). Enable logging for details.)")
        return

    while True:
        flat = render_main_menu(dataset)
        if not flat:
            print("No files found.")
            return

        choice = input("Select file: ").strip()
        if not choice or choice == "0":
            break
        if not choice.isdigit():
            print("❌ Invalid selection.")
            continue

        idx = int(choice)
        if not (1 <= idx <= len(flat)):
            print("❌ Invalid selection.")
            continue

        _, record = flat[idx - 1]
        print(render_record_details(record))
        record.plot()
        prompt_volume_and_calculate(record)


if __name__ == "__main__":
    # Example: python src/PyGUI.py "C:\Data\Rig jsons 2026-2-19"
    if len(sys.argv) > 1:
        cli_dir = Path(sys.argv[1]).expanduser()  
    else:
        cli_dir = DEFAULT_MAIN_DIR
        print(f"Using default directory {DEFAULT_MAIN_DIR}")
    
    try:
        interactive_app(cli_dir)
    except KeyboardInterrupt:
        print("\nExiting...")