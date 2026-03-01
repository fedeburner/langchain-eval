"""Utilities for writing structured harness run artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def generate_run_id(prefix: str = "memory-run") -> str:
    """Generate a stable run identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def to_jsonable(value: Any) -> Any:
    """Convert nested Python objects to JSON-safe values."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


class RunArtifactWriter:
    """Writes JSON and JSONL artifacts for one harness run."""

    def __init__(self, base_dir: str | Path, run_id: str):
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.turn_traces_path = self.run_dir / "turn_traces.jsonl"
        self.probe_results_path = self.run_dir / "probe_results.jsonl"
        self.summary_path = self.run_dir / "run_summary.json"

    def write_turn_trace(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.turn_traces_path, record)

    def write_probe_result(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.probe_results_path, record)

    def write_summary(self, payload: dict[str, Any]) -> None:
        self._write_json(self.summary_path, payload)

    @property
    def output_paths(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "turn_traces": str(self.turn_traces_path),
            "probe_results": str(self.probe_results_path),
            "run_summary": str(self.summary_path),
        }

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(to_jsonable(payload), ensure_ascii=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=True)
            fh.write("\n")
