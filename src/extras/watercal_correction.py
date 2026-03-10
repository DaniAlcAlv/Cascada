# watercal_dataset.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from models.watercal_model import WaterCalRecord

### CODE FOR REWRITING A WATERCAL JSON WITHOUT RECALCULATING THE FIT, ONLY WITH THE INTERVAL_AVG CORRECTED ###
def _has_inconsistent_interval_average(errors: List[str]) -> bool:
    prefix = "Inconsistent interval_average"
    return any(isinstance(e, str) and e.startswith(prefix) for e in errors)


def write_corrected_water_calibration_without_recalc(
    original_wc_path: Path | str,
    *,
    rewrite: bool = False,
) -> Dict[str, Any]:
    """
    Create (or update) a 'corrected_water_calibration.json' next to the given
    'water_calibration.json' recalculating the output.

    A correction is written only if the record is "not OK":
      - record.different_recalculated_output == True, OR
      - there is an "Inconsistent interval_average ..." error.

    The corrected file is a **bare** payload (no rig/computer fields).
    """

    p = Path(original_wc_path)
    report: Dict[str, Any] = {
        "file": str(p),
        "status": "skipped",
        "reason": "",
        "needs_correction": False,
        "used_interval_average": "original",
        "corrected_path": None,
    }

    if not p.exists():
        report["status"] = "error"
        report["reason"] = "file_not_found"
        return report

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"json_load_failed: {type(e).__name__}: {e}"
        return report

    # Validate → this runs WaterValveCalibration._run_checks() which caches decisions.
    try:
        rec = WaterCalRecord.model_validate(raw)
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"validation_failed_before: {type(e).__name__}: {e}"
        return report

    wv = rec.calibration.water_valve
    inconsistent = _has_inconsistent_interval_average(rec.errors)
    not_ok = bool(rec.different_recalculated_output or inconsistent)
    report["needs_correction"] = not_ok

    if not not_ok:
        report["status"] = "skipped"
        report["reason"] = "record_already_ok"
        return report

    iv = getattr(wv, "corrected_interval_average", None)

    if iv is None:
        iv = dict(sorted(wv.output.interval_average.items()))
        report["used_interval_average"] = "original"
    else:
        report["used_interval_average"] = "rebuilt" if inconsistent else "original_cached"

    # Coefficients: never recompute — use the model’s decision
    slope, offset, r2 = wv.preferred_coefficients

    bare = {
        "date": wv.date.isoformat() if wv.date else None,
        "description": wv.description,
        "notes": wv.notes,
        "input": wv.input.model_dump(mode="json"),
        "output": {
            "interval_average": iv,
            "slope": slope,
            "offset": offset,
            "r2": r2,
            "valid_domain": wv.output.valid_domain,
        },
    }

    out_dir = p.parent
    out_dir.mkdir(parents=True, exist_ok=True)  # defensive
    path = out_dir / "water_calibration.json"

    if path.exists() and not rewrite:
        report["status"] = "skipped"
        report["reason"] = f"corrected_exists: {path}"
        report["corrected_path"] = str(path)
        return report

    try:
        path.write_text(json.dumps(bare, indent=2, ensure_ascii=False), encoding="utf-8")
        report["status"] = "rewritten" if rewrite else "written"
        report["corrected_path"] = str(path)
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"write_failed: {type(e).__name__}: {e}"

    return report

def write_corrected_for_all_without_recalc(root: Path | str, *, rewrite: bool = False) -> List[Dict[str, Any]]:
    """
    Scan a directory tree for 'water_calibration.json' files and create
    'corrected_water_calibration.json' next to those that are not OK.

    Returns a list of per-file reports.
    """
    base = Path(root)
    reports: List[Dict[str, Any]] = []

    if not base.exists():
        return [{"file": str(base), "status": "error", "reason": "dir_not_found"}]

    if base.is_file():
        # If the root is actually a file, just run the single-file writer.
        return [write_corrected_water_calibration_without_recalc(base, rewrite=rewrite)]

    # Walk the tree when root is a directory
    for wc in sorted(base.rglob("*.json")):
        rep = write_corrected_water_calibration_without_recalc(wc, rewrite=rewrite)
        reports.append(rep)

    if not reports:
        reports.append({"file": str(base), "status": "skipped", "reason": "no_water_calibration_json_found"})

    return reports

# if __name__ == "__main__":
#     rep = write_corrected_water_calibration_without_recalc(r"C:\Data\Water-cal\13D_2026-01-15T004421Z")
#     print(rep)

if __name__ == "__main__":
    # Example CLI-ish invocation:
    root = r"C:\Data\water-cal-2026\water_calibration\automatic_water_calibration"  
    results = write_corrected_for_all_without_recalc(root, rewrite=True)
    for r in results:
        print(r)