"""
Water calibration browser with error detection and a calculator: Water Volume (uL) -> Solenoid open time (ms) if the volume is in the calibrated range.

Goes through a directory with this layout:
    main_dir/
        <computer_name_1>/
            config_a.json
            config_b.json
        <computer_name_2>/
            config_c.json
            ...

- And collects from each JSON file:
{
  "computer_name": str,
  "rig_name": str,           # e.g., "5A", "12A", "12B"
  "calibration": {
    "water_valve": {
      "output": {
        "interval_average": { str(float): float, ... }, 
        "slope": float,
        "offset": float,
        "r2": float
      }
    }
  }
}


List of Errors that can be detected:

- Load/Parse
  - Skipped file due to read/JSON error: "[ERROR] Skipping <file>: <exception>"
  - Schema validation failure: "[ERROR] Invalid calibration file: <file>. <ValidationError>"

- Schema/Content (Pydantic)
  - If the repeat count (n) of a mesurement is 20 or less
  - Interval_average (List with the results from dividing: measured_weigth/n for each reading) is null or not a mapping, or if it has fewer than 2 points
  - Invalid interval keys or values (negatives or non-numeric)
  - Less than 2 entries in measurements
  - Non-positive valve_open_interval, valve_open_time, or water_weight data

- Data Consistency (measurement vs. output)
  - Missing interval_average entry for a measured valve_open_time
  - Non-positive interval_average value
  - Inconsistent interval_average vs expected (from water_weight averages / repeat_count)
  - Reported vs recomputed regression mismatch (slope/offset/r2 beyond tolerance)

- Regression Quality
  - Poor fit: r2 < limit
  - Offset too large relative to slope 
  - Slope out of usual bounds (too big or small)
"""

import json
import math
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Annotated, Mapping
from pydantic import BaseModel, ValidationError, field_validator, Field

# ---------- Configuration ----------
DEFAULT_MAIN_DIR = Path(r"C:\Data\Rig jsons 2026-2-19")
MENU_WIDTH = 100 

# ---------- Regression Utilities ----------
def linear_regression(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    n = len(x)
    if n < 2:
        return None, None, None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    if ss_xx == 0:
        return None, None, None

    b = ss_xy / ss_xx
    a = mean_y - b * mean_x

    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
    return b, a, r2


# ---------- Pydantic Models ----------
PosFloat = Annotated[float, Field(gt=0)]  # Positive number


class Measurement(BaseModel):
    valve_open_interval: PosFloat
    valve_open_time: PosFloat
    water_weight: List[PosFloat]
    repeat_count: int = Field(gt=20)


class CalibrationInput(BaseModel):
    measurements: List[Measurement] = Field(min_length=2)


class RegressionResult(BaseModel):
    slope: float
    offset: float
    r2: float
    slope_ok: bool
    offset_ok: bool
    r2_ok: bool


class CalibrationOutput(BaseModel):
    interval_average: Dict[PosFloat, PosFloat]
    slope: float
    offset: float
    r2: float

    @field_validator("interval_average", mode="before")
    @classmethod
    def convert_and_validate_keys(cls, v: Any):
        if v is None:
            raise ValueError("interval_average cannot be null; expected a mapping of {open_time: weight}.")
        if not isinstance(v, Mapping):
            raise TypeError(f"interval_average must be a mapping (dict-like), got {type(v).__name__}.")

        converted: Dict[PosFloat, PosFloat] = {}

        for key, value in v.items():
            # Convert and validate key (interval/time)
            try:
                k = float(key)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid interval key: {key!r} (not a number)")
            if not math.isfinite(k) or k <= 0:
                raise ValueError(f"Invalid interval key: {key!r} (must be positive and finite)")

            # Convert and validate value (weight/volume)
            if not isinstance(value, (int, float)):
                raise TypeError(f"Weight for interval {key!r} must be a number, got {type(value).__name__}")
            val = float(value)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"Invalid weight for interval {key!r}: {value!r} (must be positive and finite)")
            converted[k] = val

        if len(converted) < 2:
            raise ValueError("interval_average must contain at least 2 points for regression.")

        return converted


class WaterValveCalibration(BaseModel):
    date: Optional[str]
    input: CalibrationInput
    output: CalibrationOutput


class Calibration(BaseModel):
    water_valve: WaterValveCalibration


# ---------- Root Model ----------
class WaterCalModel(BaseModel):
    computer_name: str
    rig_name: str
    calibration: Calibration


# ---------- Wrapper Class ----------
class WaterCalData:
    EQUAL_TOLERANCE = 0.005  # Tolerance for calculating if 2 numbers are the same or not
    MAX_OFFSET_SLOPE_RATIO = 20
    MIN_R2, MIN_R2_WARN = 0.980, 0.990
    SLOPE_MIN, SLOPE_MAX = 0.04, 0.09

    def __init__(self, file_path: Path, raw_data: Dict[str, Any]):
        try:
            self._model = WaterCalModel.model_validate(raw_data)
        except ValidationError as e:
            print(f"❌ Invalid calibration file: {file_path}. {e}")
            raise

        self.file_path = file_path
        self.rig_num: str = self._extract_rig_num(self._model.rig_name)

        self._recomputed: Optional[RegressionResult] = None
        self._warnings: set[str] = set()
        self._errors: set[str] = set()
        self._different_recalculated_output: bool = False

        self.weight_per_valve_opentime: Dict[PosFloat, PosFloat] = self.calibration_output.interval_average

        self._validate_folder_match()
        self._validate_interval_average_output()
        self._validate_regression()
        self._validate_date()

    # ---------- Public Properties ----------

    @property
    def computer_name(self) -> str:
        return self._model.computer_name

    @property
    def rig_name(self) -> str:
        return self._model.rig_name

    @property
    def date(self) -> Optional[str]:
        return self._model.calibration.water_valve.date

    @property
    def measurements(self) -> List[Measurement]:
        return self._model.calibration.water_valve.input.measurements

    @property
    def calibration_output(self) -> CalibrationOutput:
        return self._model.calibration.water_valve.output

    @property
    def recomputed(self) -> RegressionResult:
        return self._recomputed

    @property
    def preferred_coefficients(self) -> Tuple[PosFloat, PosFloat, PosFloat]:
        if self.different_recalculated_output:  # If errors exist, use recomputed
            return self._recomputed.slope, self._recomputed.offset, self._recomputed.r2
        return self.calibration_output.slope, self.calibration_output.offset, self.calibration_output.r2

    @property
    def warnings(self) -> set[str]:
        return self._warnings

    @property
    def errors(self) -> set[str]:
        return self._errors

    @property
    def n_warnings(self) -> int:
        return len(self._warnings)

    @property
    def n_errors(self) -> int:
        return len(self._errors)

    @property
    def different_recalculated_output(self) -> bool:
        return self._different_recalculated_output

    @property
    def vol_bounds(self) -> tuple[float, float]:
        keys = list(self.weight_per_valve_opentime.values())
        return (min(keys), max(keys))


    @staticmethod
    def _extract_rig_num(rig_name: Optional[str]) -> str:
        """
        Returns the leading numeric part of rig_name as a string, or 'Other' if none.
        Examples: '4C' -> '4', '12B' -> '12', 'EPHYS-5' -> 'Other'
        """
        if not rig_name:
            return "Other"
        m = re.match(r"^\s*(\d+)", rig_name)
        return m.group(1) if m else "Other"


    def __str__(self) -> str:
        warn = f" [{self.n_warnings} ⚠️ ]" if self.n_warnings else ""
        error = f" [{self.n_errors} ❌ ]" if self.n_errors else ""
        slope, offset, r2 = self.preferred_coefficients
        date_str = self.date[:10] if self.date else "NoDate"

        return (
            f"{self.file_path.name[:-5]}  "
            f"({date_str}) "
            f"W = {slope:.5f}·t +{offset:.5f} (R2={r2:.3f} )"
            f"{warn}{error}"
        )

    def __repr__(self) -> str:
        # Layout constants
        title = f" Details — {self.rig_name} @ {self.computer_name} "
        label_w = 10 # Left column label width

        # Status badges (optional; uses RegressionResult booleans)
        s_ok = "\033[92m✓\033[0m" if self.recomputed.slope_ok  else "\033[91m✗\033[0m"
        o_ok = "\033[92m✓\033[0m" if self.recomputed.offset_ok else "\033[91m✗\033[0m"
        r_ok = "\033[92m✓\033[0m" if self.recomputed.r2_ok     else "\033[91m✗\033[0m"

        # Format rows
        header   = f"┌{title:─^{MENU_WIDTH-2}}┐"
        file_row = f"│ {'File:':<{label_w}} {self.file_path}".ljust(MENU_WIDTH-1) + "│"
        date_row = f"│ {'Date:':<{label_w}} {self.date}".ljust(MENU_WIDTH-1) + "│"
        spacer   = f"├{'─'*(MENU_WIDTH-2)}" + "┤"
        footer   = f"└{'':─^{MENU_WIDTH-2}}┘"
        
        orig_row = (
            f"│ {'Original: ':<{label_w}}  "
            f"Slope {self.calibration_output.slope:>9.6f}    "
            f"Offset {self.calibration_output.offset:>9.6f}     "
            f"R² {self.calibration_output.r2:>5.3f}   "
        )
        orig_row = orig_row.ljust(MENU_WIDTH-1) + "│"

        # Recomputed coefficients + status badges
        recom_row = (
            f"│ {'Recomputed:':<{label_w}} "
            f"Slope {self.recomputed.slope:>9.6f} {s_ok}  "
            f"Offset {self.recomputed.offset:>9.6f} {o_ok}  "
            f"R² {self.recomputed.r2:>5.3f} {r_ok}   "
        )
        recom_row = recom_row.ljust(MENU_WIDTH-1) + "│"

        lines = [header, file_row, date_row, spacer, orig_row, recom_row]

        if self.warnings:
            lines.extend([spacer, f"│⚠️  WARNINGS".ljust(MENU_WIDTH) + "│"])    
            for w in self.warnings:
                lines.append((f"│  - {w}".ljust(MENU_WIDTH-1) + "│"))
        if self.errors:
            lines.extend([spacer, f"│ ❌ ERRORS".ljust(MENU_WIDTH)+ "│"])   
            for e in self.errors:
                lines.append((f"│  - {e}".ljust(MENU_WIDTH-1) + "│"))

        lines.append(footer)

        return "\n".join(lines)

    # ---------- Validation ----------
    def _validate_folder_match(self) -> None:
        folder_name = self.file_path.parent.name
        if folder_name != self._model.computer_name:
            self._warnings.add(
                f"Computer mismatch: folder='{folder_name}' vs json='{self._model.computer_name}'"
            )
    
    def _validate_date(self) -> None:
        """Validate calibration recency: warn at 4 months, error at 6 months; error if future date."""
        from datetime import datetime, timezone, timedelta
        cal_dt = datetime.fromisoformat(self.date)
        
        if not self.date:
            self._errors.add("Calibration date is missing; unable to validate recency.")
            return
        elif not cal_dt:
            self._errors.add("Calibration date is has the wrong formatting (should be  ISO 8601)")            
        
        # Use tz-aware 'now' if cal_dt is tz-aware; otherwise use naive 'now'
        if getattr(cal_dt, "tzinfo", None) is not None and cal_dt.tzinfo is not None:
            current_dt = datetime.now(timezone.utc)
            # Normalize both to UTC to avoid ambiguous comparisons
            cal_dt_utc = cal_dt.astimezone(timezone.utc)
            delta_time = current_dt - cal_dt_utc
        else:
            current_dt = datetime.now()
            delta_time = current_dt - cal_dt

        # Negative delta implies calibration date is in the future
        if delta_time.total_seconds() < 0:
            self._errors.add(
                f"Calibration date is in the future: {cal_dt.isoformat()} > {current_dt.isoformat()}"
            )
            return

        warn_period =  timedelta(days=120)  # ~4 months
        error_period = timedelta(days=180)  # ~6 months

        if delta_time > error_period:
            self._errors.add(
                f"Calibration date is too old: age={delta_time.days} > threshold ({error_period.days})."
            )
        elif delta_time > warn_period:
            self._warnings.add(
                f"Calibration date is getting old: age={delta_time.days} > threshold ({warn_period.days})."
            )

    def _validate_interval_average_output(self) -> None:
        for m in self.measurements:
            weight = sum(m.water_weight) / len(m.water_weight)  # In case there are several readings
            expected_weight = weight / m.repeat_count

            output_weight = self.weight_per_valve_opentime.get(m.valve_open_time)

            if output_weight is None:
                self._errors.add(f"Missing interval_average entry for valve_open_time={m.valve_open_time}")

            elif output_weight <= 0:
                self._errors.add(f"Non-positive interval_average value at {m.valve_open_time}: {output_weight}")

            elif math.isclose(output_weight / m.repeat_count, expected_weight, rel_tol=self.EQUAL_TOLERANCE):
                self._warnings.add("Output error: Weight was not divided by the repeat count when calculating the interval avg")
                self.weight_per_valve_opentime[m.valve_open_time] = expected_weight
                print(f'"{m.valve_open_time}": {expected_weight},')

            elif not math.isclose(output_weight, expected_weight, rel_tol=self.EQUAL_TOLERANCE):
                self._errors.add(
                    f"Inconsistent interval_average at {m.valve_open_time}: "
                    f"Expected {expected_weight:.6f}, got {output_weight:.6f}, ratio is {(expected_weight / output_weight):.2}"
                )

    def _validate_regression(self) -> None:
        x, y = map(list, zip(*sorted(self.weight_per_valve_opentime.items())))
        slope, offset, r2 = linear_regression(x, y)

        slope_ok = math.isclose(slope, self.calibration_output.slope, rel_tol=self.EQUAL_TOLERANCE)
        offset_ok = math.isclose(offset, self.calibration_output.offset, rel_tol=self.EQUAL_TOLERANCE)
        r2_ok = math.isclose(r2, self.calibration_output.r2, abs_tol=self.EQUAL_TOLERANCE/10)

        self._recomputed = RegressionResult(
            slope=slope,
            offset=offset,
            r2=r2,
            slope_ok=slope_ok,
            offset_ok=offset_ok,
            r2_ok=r2_ok,
        )

        if not (slope_ok and offset_ok and r2_ok):
            self._different_recalculated_output = True
            self._errors.add("Reported regression differs from computed.")

        # --- Quality thresholds ---
        if r2 < self.MIN_R2:
            self._errors.add(f"Poor regression fit: r2={r2:.6f} (< {self.MIN_R2})")
        elif r2 < self.MIN_R2_WARN:
            self._warnings.add(f"Suboptimal regression fit: r2={r2:.6f} (< {self.MIN_R2_WARN})")

        if abs(offset) > abs(slope) / self.MAX_OFFSET_SLOPE_RATIO:
            self._errors.add(f"Offset value ({offset:.6f}) too big compared to the the slope ({slope:.6f})")
        elif abs(offset) > abs(slope) / (self.MAX_OFFSET_SLOPE_RATIO * 5):
            self._warnings.add(f"Offset value ({offset:.6f}) quite big compared to the the slope ({slope:.6f})")

        s_range = self.SLOPE_MAX - self.SLOPE_MIN
        if self.SLOPE_MIN > slope or slope > self.SLOPE_MAX:
            self._errors.add(
                f"Slope value ({slope:.6f}) out of bounds ({self.SLOPE_MIN:.6f} - {self.SLOPE_MAX:.6f}))"
            )
        elif slope < self.SLOPE_MIN + s_range / 5 or slope > self.SLOPE_MAX - s_range / 5:
            self._warnings.add(
                f"Slope value ({slope:.6f}) close to out of bounds ({self.SLOPE_MIN:.6f} - {self.SLOPE_MAX:.6f}))"
            )

    # --- Plotting ---
    def plot_weight_vs_opentime(self, show_slope_band: bool = True, draw: bool = True):
        """
        Plot weight vs valve open time using self.weight_per_valve_opentime, with fitted regression line.

        Parameters
        ----------
        show_slope_band : bool
            If True, shades y-values implied by slope bounds to visualize acceptable range.
        draw : bool
            If True, it will show the plot on screen.

        Returns
        -------
        (fig, ax) : tuple
            Matplotlib Figure and Axes objects.
        """

        # Sort by open time
        x, y = map(list, zip(*sorted(self.weight_per_valve_opentime.items())))  # x: open time, y: weight
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Coefficients to display/use
        slope, offset, r2 = self.preferred_coefficients

        # --- Matplotlib setup
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)
        point_kwargs = dict(s=50, color="C0", edgecolor="white", linewidth=0.75, alpha=0.9, zorder=3)
        line_kwargs = dict(color="C3", lw=2.25, alpha=0.95, zorder=2)

        # --- Scatter points
        ax.scatter(x_arr, y_arr, **point_kwargs, label="Measured points")

        # --- Regression line over the data domain
        x_line = np.linspace(float(min(x_arr)), float(max(x_arr)), 200)
        y_line = offset + slope * x_line
        ax.plot(x_line, y_line, **line_kwargs, label=f"Fit: W = {slope:.5f}·t + {offset:.5f}  (R²={r2:.3f})")

        # --- Also plot the originally reported points and regression if it differs
        if self.different_recalculated_output:
            # Regression
            s0 = self.calibration_output.slope
            o0 = self.calibration_output.offset
            r20 = self.calibration_output.r2
            y_line0 = o0 + s0 * x_line
            ax.plot(
                x_line, y_line0, color="C1", lw=1.8, ls="--", alpha=0.9, 
                label=f"Original: W = {s0:.5f}·t + {o0:.5f}  (R²={r20:.3f})"
                )
            # Points from interval average
            x0, y0 = map(list, zip(*sorted(self.calibration_output.interval_average.items())))
            x0_arr = np.asarray(x0, dtype=float)
            y0_arr = np.asarray(y0, dtype=float)
            ax.scatter(
                x0_arr, y0_arr, s=40, color="C1", edgecolor="black", 
                linewidth=0.6, alpha=0.7, marker="D", label="Original interval_average"
                )

        # --- Optional slope-bounds band (visual hint only)
        if show_slope_band and math.isfinite(self.SLOPE_MIN) and math.isfinite(self.SLOPE_MAX):
            # Shade the y-range computed by applying slope bounds around the same offset.
            y_min_band = offset + self.SLOPE_MIN * x_line
            y_max_band = offset + self.SLOPE_MAX * x_line
            ax.fill_between(
                x_line, y_min_band, y_max_band,
                color="C2", alpha=0.08, label=f"Slope band [{self.SLOPE_MIN:.2f}, {self.SLOPE_MAX:.2f}]"
            )

        # --- Axis labels, title, grid and legend
        ax.set_xlabel("Valve open time (s)", fontsize=10)
        ax.set_ylabel("Water (g or mL)", fontsize=10)
        title = f"{self.file_path.stem} — Weight vs Valve Open Time"
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.35)
        ax.legend(frameon=True)

        # --- Tight layout and show
        fig.tight_layout()
        
        if draw: plt.show()

        return fig, ax

# ---------- Discovery ----------
def load_watercaldata(main_dir: Path) -> List[WaterCalData]:
    records = []
    for comp_dir in sorted(p for p in main_dir.iterdir() if p.is_dir()):
        for cfg in sorted(comp_dir.glob("*.json")):
            try:
                with cfg.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(WaterCalData(cfg, data))
            except Exception as e:
                print(f"[ERROR] Skipping {cfg}: {e}")
    return records


def rig_sort_key(rig: Optional[str]):
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", rig.strip())
    if not m:
        return (1, 999_999, rig)
    return (0, int(m.group(1)), m.group(2).upper())


def index_by_rig(records: List[WaterCalData]) -> Dict[str, List[WaterCalData]]:
    by_rig: Dict[str, List[WaterCalData]] = {}
    for r in records:
        by_rig.setdefault(r.rig_name or "(missing)", []).append(r)
    for lst in by_rig.values():
        lst.sort(key=lambda r: (r.computer_name or "", r.file_path.name))
    return by_rig


def group_records_by_rig_num(records: List[WaterCalData]) -> Dict[str, List[WaterCalData]]:
    """   Groups records by record.rig_num (e.g., '4', '12', or 'Other').   """
    groups: Dict[str, List[WaterCalData]] = {}
    for rec in records:
        key = getattr(rec, "rig_num", "Other") or "Other"
        groups.setdefault(key, []).append(rec)

    # Sort each group for stable display: by rig_name then by file name
    for lst in groups.values():
        lst.sort(key=lambda r: ((r.rig_name or ""), r.file_path.name))
    return groups


def ordered_rig_num_keys(groups: Dict[str, List[WaterCalData]]) -> List[str]:
    """  Returns group keys ordered numerically with 'Other' last. """
    numeric = [k for k in groups.keys() if k != "Other" and k.isdigit()]
    numeric_sorted = sorted(numeric, key=lambda s: int(s))
    return numeric_sorted + (["Other"] if "Other" in groups else [])


# ---------- Calculator ----------
def calc_time_ms_from_vol_ul(uL: float, slope: float, offset: float) -> float:
    """Uses the regression slope and offset to calculate the valve open time in milliseconds given a volume in microliters"""
    mL = uL/1000 # conversion because the units from the regression are t in seconds and water vol or weight in mL or g
    t_sec = (mL - offset) / slope
    t_ms = t_sec*1000
    return t_ms

# ---------- UI  ----------
def prepare_data(main_dir: Optional[Path]) -> Tuple[Path, List[WaterCalData], Dict[str, List[WaterCalData]], List[str]]:
    main_dir = main_dir or DEFAULT_MAIN_DIR
    records = load_watercaldata(main_dir)
    rigs_index = index_by_rig(records)
    rig_names = sorted(rigs_index.keys(), key=rig_sort_key)
    return main_dir, records, rigs_index, rig_names


def build_flat_menu(
    rig_names: List[str],
    rigs_index: Dict[str, List[WaterCalData]],
) -> List[Tuple[str, WaterCalData]]:
    """
    Builds a flat list of (rig_name, record) in a stable order
    for display and selection.
    """
    flat: List[Tuple[str, WaterCalData]] = []
    for rig in rig_names:
        for record in rigs_index[rig]:
            flat.append((rig, record))
    return flat


def render_main_menu_grouped_by_rig_num(records: List[WaterCalData]) -> List[Tuple[str, WaterCalData]]:
    """
    Prints a grouped (by rig_num), single-level menu and returns a flat
    list mapping (rig_num, record) for selection (1..N).
    """
    # Box layout
    title = " WATER CALIBRATIONS FOUND IN RIG JSONS"
    header = f"┌{title:─^{MENU_WIDTH-2}}┐"
    footer = f"└{'':─^{MENU_WIDTH-2}}┘"

    def row(text: str = "") -> str:
        return f"│ {text.ljust(MENU_WIDTH-3)}│"

    def section_title(rig_num: str, count: int) -> str:
        # e.g., "Rigs 4x (N files)" or "Other Rigs (N files)"
        if rig_num == "Other":
            return f"{f"├─Other Rigs───({count} file{'s' if count != 1 else ''})":─<{MENU_WIDTH-1}}┤"
        # You can choose either "Rigs 4x" or just "Rigs 4"; keeping it clean:
        return f"{f"├─Rigs {rig_num}───({count} file{'s' if count != 1 else ''})":─<{MENU_WIDTH-1}}┤"

    def entry_line(idx: int, rec: WaterCalData) -> str:
        c_lengths = [8 , 40, 10] #Max char len of the info (0:warnings 1:Formula 2:Date)
        label = rec.file_path.stem

        warn = f"{"\033[93m"}[{rec.n_warnings}!]{"\033[0m"}" if rec.n_warnings else "."*4
        err  = f"{"\033[91m"}[{rec.n_errors}!]{"\033[0m"}"   if rec.n_errors   else "."*4
        badges = f"{(warn + err) if (warn or err) else '':.^{c_lengths[0]}}"

        formula = f"W= {rec.calibration_output.slope:0.5f}·t{rec.calibration_output.offset:+0.5f}..(R2={rec.calibration_output.r2:0.3f})"
        date_str = (rec.date[:10] if rec.date else "NoDate")

        left_col = f"{idx:>4}) {label:.<{MENU_WIDTH-sum(c_lengths)-9}}"
        right_col = f"{badges}{formula:.^{c_lengths[1]}}{date_str:.>{c_lengths[2]}}"

        line = f"{left_col}{right_col}"

        return row(line)

    # Group and order
    groups = group_records_by_rig_num(records)
    ordered_keys = ordered_rig_num_keys(groups)

    # Render
    print(f"\n{header}")
    flat: List[Tuple[str, WaterCalData]] = []
    idx = 1

    for g_i, g_key in enumerate(ordered_keys):
        lst = groups[g_key]
        print(section_title(g_key, len(lst)))
        if not lst:
            print(row("  └─ (no files)"))
        else:
            for rec in lst:
                print(entry_line(idx, rec))
                flat.append((g_key, rec))
                idx += 1

    print(footer)
    print("0 (or empty) to exit")
    return flat


def select_index(prompt: str, max_index: int) -> Optional[int]:
    """
    Prompt for a number in [0..max_index]. Returns:
      - None if invalid
      - 0 for Exit
      - 1..max_index for valid selection
    """
    choice = input(prompt).strip()
    if not choice.isdigit():
        return None
    idx = int(choice)
    if idx == 0:
        return 0
    if 1 <= idx <= max_index:
        return idx
    return None


def prompt_volume_and_calculate(record: WaterCalData) -> None:
    while True:
        slope, offset, r2 = record.preferred_coefficients
        ul_str = input("Enter a volume in microliters to calculate the valve open time in milliseconds (blank or 0 to go back): ").strip()
        if not ul_str or ul_str == "0":
            break
        try:
            microliters = float(ul_str)
        except ValueError:
            print("❌ Invalid number.")
            continue

        lo, hi = record.vol_bounds
        low_vol, hi_vol = lo*1000, hi*1000
        if microliters < low_vol or microliters > hi_vol:
            print(f" => (❌ {microliters} is outside the calibrated range {low_vol}–{hi_vol} ms !)")
        else:
            print(f" => {calc_time_ms_from_vol_ul(microliters, slope, offset)} ms")
                
    print("(Back to list. Press 0 to exit.)") # small UX hint when returning to the list

def interactive_app(main_dir: Optional[Path] = None):
    main_dir = main_dir or DEFAULT_MAIN_DIR
    if not main_dir.exists():
        print(f"Path not found: {main_dir}")
        return

    records = load_watercaldata(main_dir)
    if not records:
        print("No water calibration information found.")
        return

    while True:
        flat = render_main_menu_grouped_by_rig_num(records)
        if not flat:
            print("No files found.")
            return

        choice = input("Select file: ").strip()
        if not choice or choice == "0":# Exit or go back
            break
        if not choice.isdigit():
            print("❌ Invalid selection.")
            continue

        idx = int(choice)
        if not (1 <= idx <= len(flat)): # Out-of-range number
            print("❌ Invalid selection.")
            continue

        rig_num, record = flat[idx - 1]
        print(repr(record))
        record.plot_weight_vs_opentime()
        prompt_volume_and_calculate(record)


if __name__ == "__main__":
    cli_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    try:
        interactive_app(cli_dir)
    except KeyboardInterrupt:
        print("\nExiting...")