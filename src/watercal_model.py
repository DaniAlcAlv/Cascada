# watercal_model.py
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Mapping, Annotated, ClassVar
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ---------- Regression ----------
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
    slope = ss_xy / ss_xx
    offset = mean_y - slope * mean_x
    ss_res = sum((yi - (offset + slope * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
    return slope, offset, r2


# ---------- Pydantic Models ----------
PosFloat = Annotated[float, Field(gt=0)]  # Positive number


class RegressionResult(BaseModel):
    slope: float
    offset: float
    r2: float
    slope_ok: bool
    offset_ok: bool
    r2_ok: bool


class WaterCalOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interval_average: Dict[PosFloat, PosFloat]
    slope: float
    offset: float
    r2: float
    valid_domain: Optional[List[float]] = None  # present in some files

    @field_validator("interval_average", mode="before")
    @classmethod
    def convert_and_validate_keys(cls, v: Any):
        if v is None:
            raise ValueError("interval_average cannot be null; expected a mapping of {open_time: weight}.")
        if not isinstance(v, Mapping):
            raise TypeError(f"interval_average must be a mapping (dict-like), got {type(v).__name__}.")
        converted: Dict[PosFloat, PosFloat] = {}
        for key, value in v.items():
            try:
                k = float(key)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid interval key: {key!r} (not a number)")
            if not math.isfinite(k) or k <= 0:
                raise ValueError(f"Invalid interval key: {key!r} (must be positive and finite)")
            if not isinstance(value, (int, float)):
                raise TypeError(f"Weight for interval {key!r} must be a number, got {type(value).__name__}")
            val = float(value)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"Invalid weight for interval {key!r}: {value!r} (must be positive and finite)")
            converted[k] = val
        if len(converted) < 2:
            raise ValueError("interval_average must contain at least 2 points for regression.")
        return converted


class WaterCalMeasure(BaseModel):
    model_config = ConfigDict(extra="ignore")
    valve_open_interval: PosFloat
    valve_open_time: PosFloat
    water_weight: List[PosFloat]
    repeat_count: int = Field(gt=20)  # n >= 21


class WaterCalInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    measurements: List[WaterCalMeasure] = Field(min_length=2)


class WaterValveCalibration(BaseModel):
    """
    All data checks live here:
      - interval_average vs measurements (and valid_domain if provided)
      - regression recomputation & quality thresholds
      - date recency

    Also exposes:
      - warnings, errors, recomputed, different_recalculated_output
      - convenience helpers (preferred_coefficients, check_bounds, calc...)
      - **plot(...)** method (kept!)
    """
    model_config = ConfigDict(extra="ignore")

    date: Optional[datetime]
    input: WaterCalInput
    output: WaterCalOutput
    description: Optional[str] = None
    notes: Optional[str] = None

    # Diagnostics / derived
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    recomputed: Optional[RegressionResult] = None
    different_recalculated_output: bool = False
    corrected_interval_average: Dict[float, float] = Field(default_factory=dict)


    # Thresholds (class-level constants; not serialized)
    EQUAL_TOLERANCE: ClassVar[float] = 0.005
    MAX_OFFSET_SLOPE_RATIO: ClassVar[float] = 20.0
    MIN_R2: ClassVar[float] = 0.990
    MIN_R2_WARN: ClassVar[float] = 0.995
    SLOPE_MIN: ClassVar[float] = 0.04
    SLOPE_MAX: ClassVar[float] = 0.09

    # ---- Post-parse checks ----
    def model_post_init(self, __context: Any) -> None:
        self._run_checks()

    def rerun_checks(self) -> None:
        self.warnings.clear()
        self.errors.clear()
        self.recomputed = None
        self.different_recalculated_output = False
        self._run_checks()

    # ---- Internal helpers ----
    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _error(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)

    def _run_checks(self) -> None:
        inp = self.input
        out = self.output


        # (A) valid_domain vs interval_average keys (if valid_domain provided)
        if out.valid_domain is not None:
            try:
                keys_iv = set(out.interval_average.keys())
                keys_vd = set(float(x) for x in out.valid_domain)
                if keys_iv != keys_vd:
                    self._error("Valid domain and interval_average keys don't match")
            except Exception:
                self._warn("Could not compare valid_domain vs interval_average keys (parsing issue).")


        # (B) interval_average vs measurements (strict)
        self.corrected_interval_average = dict(sorted(out.interval_average.items()))
        any_inconsistent = False
        for m in inp.measurements:
            weight_avg = sum(m.water_weight) / len(m.water_weight)
            expected = weight_avg / m.repeat_count
            out_w = out.interval_average.get(m.valve_open_time)
            if out_w is None:
                self._error(f"Missing interval_average entry for valve_open_time={m.valve_open_time}")
                any_inconsistent = True
                continue
            if out_w <= 0:
                self._error(f"Non-positive interval_average value at {m.valve_open_time}: {out_w}")
                any_inconsistent = True
                continue
            if not math.isclose(out_w, expected, rel_tol=self.EQUAL_TOLERANCE):
                ratio = expected / (out_w if out_w != 0 else float('inf'))
                self._error(
                    f"Inconsistent interval_average at {m.valve_open_time}: "
                    f"expected {expected:.6f}, got {out_w:.6f}, ratio={ratio:.3f}"
                )
                any_inconsistent = True

        # If inconsistent, rebuild once 
        if any_inconsistent:
            rebuilt: Dict[float, float] = {}
            for m in inp.measurements:
                avg = sum(m.water_weight) / len(m.water_weight)
                rebuilt[m.valve_open_time] = avg / m.repeat_count
            self.corrected_interval_average = dict(sorted(rebuilt.items()))


        # (C) Regression recomputation & quality thresholds
        xs, ys = map(list, zip(*sorted(out.interval_average.items())))
        slope, offset, r2 = linear_regression(xs, ys)
        if slope is None or offset is None or r2 is None:
            self._error("Unable to compute regression (insufficient or invalid data).")
            slope, offset, r2 = float("nan"), float("nan"), float("nan")

        slope_ok = math.isclose(slope, out.slope, rel_tol=self.EQUAL_TOLERANCE) if math.isfinite(slope) else False
        offset_ok = math.isclose(offset, out.offset, rel_tol=self.EQUAL_TOLERANCE) if math.isfinite(offset) else False
        r2_ok    = math.isclose(r2, out.r2, abs_tol=self.EQUAL_TOLERANCE/10) if math.isfinite(r2) else False

        self.recomputed = RegressionResult(
            slope=slope, offset=offset, r2=r2,
            slope_ok=slope_ok, offset_ok=offset_ok, r2_ok=r2_ok
        )
        self.different_recalculated_output = not (slope_ok and offset_ok and r2_ok)

        if math.isfinite(r2):
            if r2 < self.MIN_R2:
                self._error(f"Poor regression fit: R²={r2:.6f} (< {self.MIN_R2})")
            elif r2 < self.MIN_R2_WARN:
                self._warn(f"Suboptimal regression fit: R²={r2:.6f} (< {self.MIN_R2_WARN})")

        if math.isfinite(slope) and math.isfinite(offset):
            if abs(offset) > abs(slope) / self.MAX_OFFSET_SLOPE_RATIO:
                self._error(f"Offset value ({offset:.6f}) too big compared to the the slope ({slope:.6f})")
            elif abs(offset) > abs(slope) / (self.MAX_OFFSET_SLOPE_RATIO * 5):
                self._warn(f"Offset value ({offset:.6f}) quite big compared to the the slope ({slope:.6f})")

            if slope < self.SLOPE_MIN or slope > self.SLOPE_MAX:
                self._error(f"Slope value ({slope:.6f}) out of bounds ({self.SLOPE_MIN:.6f} - {self.SLOPE_MAX:.6f})")
            else:
                span = self.SLOPE_MAX - self.SLOPE_MIN
                if slope < self.SLOPE_MIN + span / 5 or slope > self.SLOPE_MAX - span / 5:
                    self._warn(f"Slope value ({slope:.6f}) close to out of bounds ({self.SLOPE_MIN:.6f} - {self.SLOPE_MAX:.6f})")

        # (D) Date recency
        dt = self.date
        if dt is None:
            self._error("Calibration date is missing; unable to validate recency.")
        else:
            if dt.tzinfo is not None:
                now = datetime.now(timezone.utc)
                delta = now - dt.astimezone(timezone.utc)
            else:
                now = datetime.now()
                delta = now - dt
            if delta.total_seconds() < 0:
                self._error(f"Calibration date is in the future: {dt.isoformat()} > {now.isoformat()}")
            else:
                warn_period = timedelta(days=121)   # ~4 months
                error_period = timedelta(days=182)  # ~6 months
                if delta > error_period:
                    self._error(f"Calibration date is too old: age={delta.days}days > threshold={error_period.days}days.")
                elif delta > warn_period:
                    self._warn(f"Calibration date is getting old: age={delta.days}days > threshold={warn_period.days}days.")

    # ---- Convenience API on the calibration itself ----
    @property
    def preferred_coefficients(self) -> Tuple[float, float, float]:
        if self.different_recalculated_output and self.recomputed is not None:
            return self.recomputed.slope, self.recomputed.offset, self.recomputed.r2
        return self.output.slope, self.output.offset, self.output.r2

    def check_bounds(self, value: float, unit: str) -> tuple[bool, float, float]:
        vals_ml = list(self.output.interval_average.values())
        if not vals_ml:
            return False, float("nan"), float("nan")
        vals = [v * 1000.0 for v in vals_ml] if unit == "microliters" else vals_ml
        lo, hi = min(vals), max(vals)
        return (lo <= value <= hi, lo, hi)

    def calc_milliseconds_from_microliters(self, uL: float) -> float:
        s, o, _ = self.preferred_coefficients
        t_sec = ((uL / 1000.0) - o) / s
        return t_sec * 1000.0

    # ---- Plotting (kept here) ----
    def plot(
        self,
        show_slope_band: bool = True,
        draw: bool = True,
        size: tuple[float, float] = (6.0, 3.0),
        title: Optional[str] = None,
    ):
        """
        Plot weight vs valve open time using output.interval_average, with fitted regression line.

        Parameters
        ----------
        show_slope_band : bool
            If True, shade the range implied by slope bounds to visualize acceptable range.
        draw : bool
            If True, call plt.show().
        size : (w, h) inches
        title : Optional[str]
            Plot title. If None, uses a generic title.

        Returns
        -------
        (fig, ax) : tuple
            Matplotlib Figure and Axes objects.
        """
        iv = self.output.interval_average
        if not iv:
            raise ValueError("Nothing to plot: interval_average is empty.")

        # Sort by open time
        x, y = map(list, zip(*sorted(iv.items())))
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        slope, offset, r2 = self.preferred_coefficients

        fig, ax = plt.subplots(figsize=size)
        point_kwargs = dict(s=50, color="C0", edgecolor="white", linewidth=0.75, alpha=0.9, zorder=3)
        line_kwargs = dict(color="C3", lw=2.25, alpha=0.95, zorder=2)

        # Scatter points
        ax.scatter(x_arr, y_arr, **point_kwargs, label="Measured points")

        # Regression line over the domain
        x_line = np.linspace(float(min(x_arr)), float(max(x_arr)), 200)
        y_line = offset + slope * x_line
        ax.plot(x_line, y_line, **line_kwargs, label=f"Fit: W = {slope:.5f}·t {offset:+.5f}  (R²={r2:.3f})")

        # Also plot the reported regression if it differs
        if self.different_recalculated_output and self.recomputed is not None:
            s0 = self.output.slope
            o0 = self.output.offset
            r20 = self.output.r2
            y_line0 = o0 + s0 * x_line
            ax.plot(
                x_line, y_line0,
                color="C1", lw=1.8, ls="--", alpha=0.9,
                label=f"Original: W = {s0:.5f}·t {o0:+.5f}  (R²={r20:.3f})"
            )
            # Optional: re-scatter original points differently (same points here)
            ax.scatter(
                x_arr, y_arr, s=40, color="C1", edgecolor="black",
                linewidth=0.6, alpha=0.7, marker="D", label="Reported interval_average"
            )

        # Optional slope bounds band
        if show_slope_band and math.isfinite(self.SLOPE_MIN) and math.isfinite(self.SLOPE_MAX):
            y_min_band = offset + self.SLOPE_MIN * x_line
            y_max_band = offset + self.SLOPE_MAX * x_line
            ax.fill_between(
                x_line, y_min_band, y_max_band,
                color="C2", alpha=0.08, label=f"Slope band [{self.SLOPE_MIN:.2f}, {self.SLOPE_MAX:.2f}]"
            )

        ax.set_xlabel("Valve open time (s)", fontsize=8)
        ax.set_ylabel("Water (g or mL)", fontsize=8)
        ax.set_title(title or "Weight vs Valve Open Time", fontsize=9, pad=10)
        ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.35)
        ax.legend(frameon=True)

        fig.tight_layout()
        if draw:
            plt.show()
        return fig, ax


class Calibration(BaseModel):
    model_config = ConfigDict(extra="ignore")
    water_valve: WaterValveCalibration


class WaterCalRecord(BaseModel):
    """  Envelope that holds the calibration and identifying rig info  """
    model_config = ConfigDict(extra="ignore")

    computer_name: str
    rig_name: str
    calibration: Calibration
    record_id: Optional[str] = "NoNameFound"

    # ---- Optional: accept bare 'water_valve' payload and wrap it ----
    @model_validator(mode="before")
    @classmethod
    def wrap_bare_water_valve_if_needed(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if {"computer_name", "rig_name", "calibration"}.issubset(data.keys()):
            return data
        if "input" in data and "output" in data and "calibration" not in data:
            computer = data.pop("_inferred_computer_name", "(unknown-computer)")
            rig = data.pop("_inferred_rig_name", "(standalone)")
            return {
                "computer_name": computer,
                "rig_name": rig,
                "calibration": {"water_valve": {"date": data.get("date"), "input": data["input"], "output": data["output"]}},
            }
        return data

    # ---- Thin proxies for UI convenience ----
    @property
    def date(self) -> Optional[datetime]:
        return self.calibration.water_valve.date

    @property
    def warnings(self) -> List[str]:
        return self.calibration.water_valve.warnings

    @property
    def errors(self) -> List[str]:
        return self.calibration.water_valve.errors
    
    @property
    def n_errors(self) -> int:
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    @property
    def recomputed(self) -> Optional[RegressionResult]:
        return self.calibration.water_valve.recomputed

    @property
    def different_recalculated_output(self) -> bool:
        return self.calibration.water_valve.different_recalculated_output

    @property
    def preferred_coefficients(self) -> Tuple[float, float, float]:
        return self.calibration.water_valve.preferred_coefficients

    @property
    def cal_output(self) -> WaterCalOutput:
        return self.calibration.water_valve.output

    def check_bounds(self, value: float, unit: str) -> tuple[bool, float, float]:
        return self.calibration.water_valve.check_bounds(value, unit)

    def calc_milliseconds_from_microliters(self, uL: float) -> float:
        return self.calibration.water_valve.calc_milliseconds_from_microliters(uL)

    # ---- Keep a `.plot(...)` on the envelope as well (delegates) ----
    def plot(
        self,
        show_slope_band: bool = True,
        draw: bool = True,
        size: tuple[float, float] = (6.0, 3.0),
        title: Optional[str] = None,
    ):
        title = title or f"{self.rig_name} @ {self.computer_name} — Weight vs Valve Open Time"
        return self.calibration.water_valve.plot(
            show_slope_band=show_slope_band, draw=draw, size=size, title=title
        )