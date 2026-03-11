# watercal_dataset.py
from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.watercal_model import WaterCalRecord

logger = logging.getLogger()


def rig_sort_key(rig: Optional[str]):
    """Sort key: (is_other, numeric_part, letter_part) so '12B' < '12C' < 'Other'."""
    if not rig:
        return (1, 999_999, "")
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", rig.strip())
    if not m:
        return (1, 999_999, rig)
    return (0, int(m.group(1)), m.group(2).upper())


@dataclass
class WaterCalDataset:
    """
    Loads and indexes WaterCalRecord items from rig JSONs or bare watercal JSONs.

    Automatically infers rig_name/computer_name when missing.
    Provides indexing, filtering, and summaries.
    """
    main_dir: Path
    records: List[WaterCalRecord] = field(default_factory=list)
    _by_rig_name: Dict[str, List[WaterCalRecord]] = field(default_factory=dict, init=False, repr=False)
    skipped_files: List[Tuple[Path, str]] = field(default_factory=list, init=False)

    # ---------- Construction ----------
    @classmethod
    def load_from_rigs(cls, main_dir: Path | str) -> WaterCalDataset:
        main_dir = Path(main_dir)
        records: List[WaterCalRecord] = []
        skipped: List[Tuple[Path, str]] = []

        if not main_dir.exists():
            logger.warning("Main directory not found: %s", main_dir)
            dataset = cls(main_dir=main_dir)
            dataset.skipped_files = skipped
            return dataset

        # Expect main_dir/<computer>/*.json
        for comp_dir in sorted(p for p in main_dir.iterdir() if p.is_dir()):
            for rig_json_path in sorted(comp_dir.glob("*.json")):
                logger.debug("Loading rig metadata from ", rig_json_path)
                try:
                    with rig_json_path.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                    rec = WaterCalRecord.model_validate(raw)
                    rec.file_path = rig_json_path
                    records.append(rec)
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"
                    skipped.append((rig_json_path, reason))
                    logger.warning("Skipping %s — %s", rig_json_path, reason)

        dataset = cls(main_dir=main_dir, records=records)
        dataset.skipped_files = skipped
        dataset._build_indexes()
        logger.info(
            "Loaded %d records from %s (skipped %d)",
            len(records), main_dir, len(skipped)
        )
        return dataset

    @classmethod
    def load_from_water_cal_dir(cls, main_path: Path | str) -> "WaterCalDataset":
        """
        Scan a directory tree for water-calibration folders:
          Dir/(optional subdirs)/
              1) water_calibration.json
              2A) behavior/Logs/rig_input.json   (for rig metadata in automatic calibrations)
              2B) rig_info.json (alternative metadata location for manual calibrations)

        Returns:
            WaterCalDataset with WaterCalRecord items loaded and indexed.
        """

        def _get_rig_metadata(metadata_path: Path):
            try:
                with metadata_path.open("r", encoding="utf-8") as rf:
                    rig_meta = json.load(rf)
                inferred_computer = rig_meta.get("computer_name")
                inferred_rig = rig_meta.get("rig_name")
                return inferred_computer, inferred_rig
            except Exception as e:
                logger.debug("Failed reading %s: %s", rig_input_path, e)
                return "(unknown-computer)", "(unknown-rig)"

        main_dir = Path(main_path)
        records: List[WaterCalRecord] = []
        skipped: List[Tuple[Path, str]] = []

        if not main_dir.exists():
            logger.warning("Main directory not found: %s", main_dir)
            ds = cls(main_dir=main_dir)
            ds.skipped_files = skipped
            return ds

        # Find all water_calibration.json anywhere under main_dir
        wc_files = sorted(main_dir.rglob("water_calibration.json"))

        for wc_path in wc_files:
            logger.debug("Loading automatic calibration file", wc_path)
            try:
                # Load the water_calibration payload
                with wc_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)

                # Try to read behavior/Logs/rig_input.json for metadata
                rig_dir = wc_path.parent  # <rig_name>_<date>
                rig_input_path = rig_dir / "behavior" / "Logs" / "rig_input.json"
                rig_info_path = wc_path.parent / "rig_info.json"
                inferred_computer = "(unknown-computer)"
                inferred_rig = "(unknown-rig)"

                if rig_input_path.exists():
                    inferred_computer, inferred_rig = _get_rig_metadata(rig_input_path)
                elif rig_info_path.exists():
                    inferred_computer, inferred_rig = _get_rig_metadata(rig_info_path)

                # Provide inferred names for the WaterCalRecord "bare" case
                # (these keys are honored by WaterCalRecord.wrap_bare_water_valve_if_needed)
                raw.setdefault("_inferred_computer_name", inferred_computer)
                raw.setdefault("_inferred_rig_name", inferred_rig)

                # Build the record (works both for bare 'input/output' and envelope)
                rec = WaterCalRecord.model_validate(raw)
                rec.file_path = wc_path

                records.append(rec)

            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append((wc_path, reason))
                logger.warning("Skipping %s — %s", wc_path, reason)


        old_wc_files = sorted(main_dir.rglob("*water_valve_calibration*.json"))
        for old_wc_path in old_wc_files:
            logger.debug("Loading old file", old_wc_path)
            try:
                # Load the water_calibration payload
                with old_wc_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)

                # Try to read behavior/Logs/rig_input.json for metadata
                inferred_computer = "(unknown-computer)"
                try:
                    inferred_rig = old_wc_path.stem.split("_")[0]
                except:
                    inferred_rig = "(unknown-rig)"

                # Provide inferred names for the WaterCalRecord "bare" case
                # (these keys are honored by WaterCalRecord.wrap_bare_water_valve_if_needed)
                raw.setdefault("_inferred_computer_name", inferred_computer)
                raw.setdefault("_inferred_rig_name", inferred_rig)

                # Build the record (works both for bare 'input/output' and envelope)
                rec = WaterCalRecord.model_validate(raw)
                rec.file_path = wc_path

                records.append(rec)

            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                skipped.append((wc_path, reason))
                logger.warning("Skipping %s — %s", wc_path, reason)

        ds = cls(main_dir=main_dir, records=records)
        ds.skipped_files = skipped
        ds._build_indexes()
        logger.info(
            "Loaded %d records from %s (skipped %d)",
            len(records), main_dir, len(skipped)
        )
        return ds


    # ---------- Indexing ----------
    def _build_indexes(self) -> None:
        self._by_rig_name.clear()

        for r in self.records:
            self._by_rig_name.setdefault(r.rig_name or "(missing)", []).append(r)

        # Sort lists within indexes
        for lst in self._by_rig_name.values():
            lst.sort(key=lambda r: (r.rig_name or "", r.file_path))


    # ---------- Accessors ----------
    def by_rig_name(self) -> Dict[str, List[WaterCalRecord]]:
        return {k: list(v) for k, v in self._by_rig_name.items()}


    # ---------- Filters ----------
    def all(self) -> List[WaterCalRecord]:
        return list(self.records)

    def valid(self) -> List[WaterCalRecord]:
        return [r for r in self.records if r.n_errors == 0]

    def with_errors(self) -> List[WaterCalRecord]:
        return [r for r in self.records if r.n_errors > 0]

    def with_warnings(self) -> List[WaterCalRecord]:
        return [r for r in self.records if r.n_warnings > 0]

    def for_rig_name(self, rig_name: str) -> List[WaterCalRecord]:
        return list(self._by_rig_name.get(rig_name, []))

    def recent_only(self, max_age_days: int = 180) -> List[WaterCalRecord]:
        cutoff_naive = datetime.now()
        cutoff_aware = datetime.now(timezone.utc)

        out: List[WaterCalRecord] = []
        for r in self.records:
            d = r.date
            if not d:
                continue
            if d.tzinfo is None:
                age = cutoff_naive - d
            else:
                age = cutoff_aware - d.astimezone(timezone.utc)
            if age <= timedelta(days=max_age_days):
                out.append(r)
        return out