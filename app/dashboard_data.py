"""Data loading helpers for the memory dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


RUNS_ROOT = Path("evals/runs")

SCENARIO_DISPLAY_NAMES: dict[str, str] = {
    "recency_vs_distance": "Long-Range Recall",
    "conflict_resolution": "Conflict Resolution",
    "compositional_recall": "Compositional Recall",
    "noise_robustness": "Noise Robustness",
    "instruction_preference_persistence": "Style Preference",
    "cross_session_longterm": "Cross-Session Memory",
    "deep_conversation": "Deep Conversation",
    "repeated_updates": "Repeated Updates",
}

SCENARIO_SUBTITLES: dict[str, str] = {
    "recency_vs_distance": "Can it remember facts from the start after many filler turns?",
    "conflict_resolution": "Does it use the latest value when a fact is corrected?",
    "compositional_recall": "Can it combine facts given in separate turns?",
    "noise_robustness": "Do important facts survive heavy irrelevant chatter?",
    "instruction_preference_persistence": "Does it remember and apply style instructions?",
    "cross_session_longterm": "Are facts retained after a session break?",
    "deep_conversation": "Can it recall details from 20+ turns ago?",
    "repeated_updates": "When a fact changes 3 times, does it track the latest?",
}

SCENARIO_ORDER = list(SCENARIO_DISPLAY_NAMES.keys())


def scenario_label(raw: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


def scenario_subtitle(raw: str) -> str:
    return SCENARIO_SUBTITLES.get(raw, "")


def strategy_label(row: dict[str, Any]) -> str:
    return str(row.get("strategy_name") or row.get("strategy", "unknown"))


def list_available_runs(runs_root: Path = RUNS_ROOT) -> list[Path]:
    if not runs_root.exists():
        return []
    runs = [
        child
        for child in runs_root.iterdir()
        if child.is_dir() and (child / "run_summary.json").exists()
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_run(run_dir: str) -> dict[str, Any]:
    base = Path(run_dir)
    summary = _read_json(base / "run_summary.json")
    turns = pd.DataFrame(_read_jsonl(base / "turn_traces.jsonl"))
    probes = pd.DataFrame(_read_jsonl(base / "probe_results.jsonl"))

    if not turns.empty:
        for col in ("replicate_idx", "turn_idx"):
            if col in turns.columns:
                turns[col] = pd.to_numeric(turns[col], errors="coerce").fillna(0).astype(int)
        turns = turns.sort_values(["replicate_idx", "scenario", "strategy", "turn_idx"]).reset_index(drop=True)

    return {
        "summary": summary,
        "results": pd.DataFrame(summary.get("results", [])),
        "turns": turns,
        "probes": probes,
    }


def get_conversation(turns: pd.DataFrame, scenario: str, strategy: str) -> pd.DataFrame:
    if turns.empty:
        return pd.DataFrame()
    mask = (turns["scenario"] == scenario) & (turns["strategy"] == strategy)
    if "replicate_idx" in turns.columns:
        first_rep = turns.loc[mask, "replicate_idx"].min()
        mask = mask & (turns["replicate_idx"] == first_rep)
    return turns[mask].sort_values("turn_idx").reset_index(drop=True)


def get_probes_for(probes: pd.DataFrame, scenario: str, strategy: str) -> pd.DataFrame:
    if probes.empty:
        return pd.DataFrame()
    mask = (probes["scenario"] == scenario) & (probes["strategy"] == strategy)
    if "replicate_idx" in probes.columns:
        first_rep = probes.loc[mask, "replicate_idx"].min()
        mask = mask & (probes["replicate_idx"] == first_rep)
    return probes[mask].reset_index(drop=True)
