from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class M3EvidenceSample:
    scenario_id: str
    failure_mode: str
    metadata: dict[str, Any]
    summary: dict[str, Any]
    sample_dir: Path
    arrays_path: Path | None = None

    def ratio(self, name: str) -> float:
        return float(self.summary.get(name, 0.0))

    def count(self, name: str) -> int:
        return int(self.summary.get(name, 0))

    def load_arrays(self) -> dict[str, np.ndarray]:
        if self.arrays_path is None:
            raise FileNotFoundError(f"sample has no arrays.npz path: {self.scenario_id}")
        if not self.arrays_path.exists():
            raise FileNotFoundError(f"missing sample arrays: {self.arrays_path}")
        with np.load(self.arrays_path) as archive:
            return {key: archive[key] for key in archive.files}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_m3_capture(capture_dir: str | Path) -> list[M3EvidenceSample]:
    root = Path(capture_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing M3 capture manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    samples: list[M3EvidenceSample] = []
    for item in manifest.get("scenarios", []):
        scenario_id = str(item["scenario_id"])
        metadata_path = root / str(item["metadata_path"])
        summary_path = root / str(item["evidence_summary_path"])
        arrays_path = root / str(item["arrays_path"]) if "arrays_path" in item else None
        metadata = _read_json(metadata_path)
        summary = _read_json(summary_path)
        samples.append(
            M3EvidenceSample(
                scenario_id=scenario_id,
                failure_mode=str(metadata.get("failure_mode", summary.get("failure_mode", "unknown"))),
                metadata=metadata,
                summary=summary,
                sample_dir=metadata_path.parent,
                arrays_path=arrays_path,
            )
        )

    if not samples:
        raise ValueError(f"M3 capture has no samples: {capture_dir}")
    return samples


def latest_m3_capture(root: str | Path = "data/m3_capture") -> Path:
    root_path = Path(root)
    candidates = sorted(path for path in root_path.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no M3 captures under {root_path}")
    return candidates[-1]
