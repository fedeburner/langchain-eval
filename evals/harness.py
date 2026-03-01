"""Memory strategy benchmark harness with artifacts and adjudicated scoring."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent.core import make_agent
from agent.memory import (
    STRATEGY_NAMES,
    FullBufferMemory,
    HybridMemory,
    LongTermProfileMemory,
    ProfileMemory,
    SummaryMemory,
    WindowMemory,
    _to_lc_messages,
    msg_content,
)
from evals.reporting import RunArtifactWriter, generate_run_id
from evals.scenarios import SCENARIOS, Scenario, Turn, get_scenario


@dataclass
class ProbeResult:
    run_id: str
    replicate_idx: int
    scenario: str
    competency: str
    strategy: str
    strategy_name: str
    label: str
    expected: str
    expected_keywords: list[str]
    response: str
    raw_judge_result: bool
    raw_judge_rationale: str
    deterministic_match: bool
    final_adjudicated_result: bool
    adjudication_method: str
    adjudication_note: str
    latency_s: float
    messages_sent: int
    tokens_sent_est: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class TrialReport:
    run_id: str
    replicate_idx: int
    scenario_name: str
    scenario_competency: str
    strategy_slug: str
    strategy_name: str
    strategy_desc: str
    failure_interpretation: str
    probe_results: list[ProbeResult] = field(default_factory=list)
    total_latency_s: float = 0.0
    total_invoke_calls: int = 0
    total_tokens_sent_est: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    session_break_count: int = 0

    @property
    def probe_count(self) -> int:
        return len(self.probe_results)

    @property
    def raw_pass_count(self) -> int:
        return sum(1 for row in self.probe_results if row.raw_judge_result)

    @property
    def final_pass_count(self) -> int:
        return sum(1 for row in self.probe_results if row.final_adjudicated_result)

    @property
    def disagreement_count(self) -> int:
        return sum(
            1 for row in self.probe_results if row.raw_judge_result != row.final_adjudicated_result
        )


_JUDGE_MODELS: dict[str, Any] = {}


def _get_judge_model(model_str: str):
    if model_str not in _JUDGE_MODELS:
        _JUDGE_MODELS[model_str] = init_chat_model(model_str)
    return _JUDGE_MODELS[model_str]


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9+#.\- ]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_keywords(expected: str, explicit_keywords: tuple[str, ...]) -> list[str]:
    if explicit_keywords:
        return [kw.strip() for kw in explicit_keywords if kw.strip()]
    chunks = re.split(r",| and ", expected)
    keywords = [chunk.strip() for chunk in chunks if chunk.strip()]
    return keywords if keywords else [expected.strip()]


def _deterministic_match(response: str, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return False
    norm_response = _normalize_text(response)
    for keyword in expected_keywords:
        if _normalize_text(keyword) not in norm_response:
            return False
    return True


def _parse_judge_json(raw_text: str) -> tuple[bool, str]:
    json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            verdict = str(payload.get("verdict", "")).upper()
            rationale = str(payload.get("rationale", "")).strip()
            return verdict == "PASS", rationale
        except json.JSONDecodeError:
            pass
    line = raw_text.upper()
    if "PASS" in line and "FAIL" not in line:
        return True, "Judge response parsed with PASS fallback."
    if "FAIL" in line:
        return False, "Judge response parsed with FAIL fallback."
    return False, "Judge output was not parseable; defaulted to FAIL."


def _judge_recall(response: str, expected: str, judge_model_str: str) -> tuple[bool, str]:
    prompt = (
        "You are a strict evaluation judge.\n"
        "Determine if RESPONSE satisfies EXPECTED facts with semantic equivalence.\n"
        "If RESPONSE explicitly says it does not know, verdict must be FAIL.\n\n"
        f"EXPECTED:\n{expected}\n\n"
        f"RESPONSE:\n{response}\n\n"
        'Return only valid JSON: {"verdict":"PASS|FAIL","rationale":"<short reason>"}'
    )
    result = _get_judge_model(judge_model_str).invoke([HumanMessage(content=prompt)])
    return _parse_judge_json(msg_content(result).strip())


def _load_overrides(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"Override file not found: {path}")

    with override_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    base = payload.get("overrides", payload)
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            normalized[key] = {
                "result": bool(value.get("result", False)),
                "reason": str(value.get("reason", "manual override")),
            }
        else:
            normalized[key] = {"result": bool(value), "reason": "manual override"}
    return normalized


def _override_for_probe(
    overrides: dict[str, dict[str, Any]],
    scenario_name: str,
    strategy_slug: str,
    probe_label: str,
) -> dict[str, Any] | None:
    keys = [
        f"{scenario_name}:{strategy_slug}:{probe_label}",
        f"{strategy_slug}:{probe_label}",
        probe_label,
    ]
    for key in keys:
        if key in overrides:
            return overrides[key]
    return None


def _adjudicate_probe(
    *,
    scenario_name: str,
    strategy_slug: str,
    probe_label: str,
    response: str,
    expected: str,
    expected_keywords: tuple[str, ...],
    judge_model_str: str,
    overrides: dict[str, dict[str, Any]],
) -> tuple[bool, str, bool, bool, str, str]:
    raw_judge_result, raw_judge_rationale = _judge_recall(
        response=response,
        expected=expected,
        judge_model_str=judge_model_str,
    )
    keywords = _extract_keywords(expected, expected_keywords)
    deterministic = _deterministic_match(response, keywords)

    final_result = raw_judge_result
    adjudication_method = "llm_judge"
    adjudication_note = raw_judge_rationale

    if deterministic and not raw_judge_result:
        final_result = True
        adjudication_method = "deterministic_override"
        adjudication_note = (
            "Deterministic keyword match overruled judge fail; likely judge false negative."
        )

    override = _override_for_probe(overrides, scenario_name, strategy_slug, probe_label)
    if override is not None:
        final_result = bool(override["result"])
        adjudication_method = "manual_override"
        adjudication_note = str(override.get("reason", "manual override"))

    return (
        raw_judge_result,
        raw_judge_rationale,
        deterministic,
        final_result,
        adjudication_method,
        adjudication_note,
    )


def _serialize_messages_for_trace(messages: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for msg in messages:
        role = "unknown"
        if isinstance(msg, dict):
            role = str(msg.get("role", "unknown"))
        else:
            role = type(msg).__name__.replace("Message", "").lower()
        content = msg_content(msg)
        rows.append(
            {
                "role": role,
                "content_preview": content[:320],
                "content_chars": len(content),
            }
        )
    return rows


def _latest_assistant_message(result_messages: list[Any]) -> tuple[Any | None, str]:
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage):
            text = msg_content(msg).strip()
            if text:
                return msg, text
    if result_messages:
        msg = result_messages[-1]
        return msg, msg_content(msg).strip()
    return None, ""


def _extract_usage_tokens(ai_msg: Any | None) -> dict[str, int | None]:
    if ai_msg is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    usage = getattr(ai_msg, "usage_metadata", None)
    if not isinstance(usage, dict):
        usage = None

    if usage is None:
        response_meta = getattr(ai_msg, "response_metadata", {})
        if isinstance(response_meta, dict):
            candidate = response_meta.get("usage") or response_meta.get("token_usage")
            if isinstance(candidate, dict):
                usage = candidate

    if not isinstance(usage, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    def _pick(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
        return None

    return {
        "input_tokens": _pick("input_tokens", "prompt_tokens"),
        "output_tokens": _pick("output_tokens", "completion_tokens"),
        "total_tokens": _pick("total_tokens"),
    }


def _estimate_turn_cost(input_tokens: int, output_tokens: int, args) -> float:
    return (
        (input_tokens / 1000.0) * args.input_cost_per_1k
        + (output_tokens / 1000.0) * args.output_cost_per_1k
    )


def _build_strategy(slug: str, args, scenario_name: str, replicate_idx: int) -> Any:
    if slug == "full":
        return FullBufferMemory()
    if slug == "window":
        return WindowMemory(k=args.window_k)
    if slug == "summary":
        return SummaryMemory(
            model_str=args.model,
            max_tokens=args.summary_max_tokens,
            keep_recent=args.summary_keep_recent,
        )
    if slug == "profile":
        return ProfileMemory(k=args.window_k)
    if slug == "hybrid":
        return HybridMemory(
            k=args.window_k,
            model_str=args.model,
            max_tokens=args.summary_max_tokens,
            keep_recent=args.summary_keep_recent,
        )
    if slug == "longterm_profile":
        memory_key = f"{args.memory_key}:{scenario_name}:rep{replicate_idx}"
        return LongTermProfileMemory(
            k=args.window_k,
            memory_store_path=args.longterm_store_path,
            memory_key=memory_key,
        )
    raise ValueError(f"Unsupported strategy {slug!r}")


def run_trial(
    *,
    run_id: str,
    replicate_idx: int,
    scenario: Scenario,
    strategy_slug: str,
    args,
    writer: RunArtifactWriter,
    overrides: dict[str, dict[str, Any]],
) -> TrialReport:
    """Run one scenario with one strategy in one replicate."""
    strategy = _build_strategy(strategy_slug, args, scenario.name, replicate_idx)
    agent = make_agent(args.model)
    report = TrialReport(
        run_id=run_id,
        replicate_idx=replicate_idx,
        scenario_name=scenario.name,
        scenario_competency=scenario.competency,
        strategy_slug=strategy_slug,
        strategy_name=strategy.name,
        strategy_desc=strategy.description,
        failure_interpretation=scenario.failure_interpretation,
    )

    if args.verbose:
        print(f"\n{'=' * 74}")
        print(
            f"Replicate={replicate_idx + 1} | Scenario={scenario.name} | "
            f"Strategy={strategy_slug} ({strategy.name})"
        )
        print(f"{'=' * 74}")

    for turn_idx, turn in enumerate(scenario.turns, start=1):
        if turn.is_session_break():
            report.session_break_count += 1
            strategy.start_new_session()
            if args.verbose:
                print(f"[{turn_idx:02d}] SESSION_BREAK")
            writer.write_turn_trace(
                {
                    "run_id": run_id,
                    "replicate_idx": replicate_idx,
                    "scenario": scenario.name,
                    "competency": scenario.competency,
                    "strategy": strategy_slug,
                    "strategy_name": strategy.name,
                    "turn_idx": turn_idx,
                    "turn_kind": turn.kind.value,
                    "user_message": turn.user_message,
                    "assistant_response": "",
                    "messages_sent_count": 0,
                    "tokens_sent_est": 0,
                    "latency_s": 0.0,
                    "memory_snapshot": strategy.snapshot(),
                    "messages_sent_preview": [],
                    "probe_label": turn.probe_label,
                    "expected": turn.expected,
                    "judge_result": None,
                    "judge_rationale": None,
                    "deterministic_match": None,
                    "final_adjudicated_result": None,
                    "adjudication_method": None,
                    "adjudication_note": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            )
            continue

        strategy.add_user_message(turn.user_message)
        messages_to_send = strategy.get_messages()
        memory_snapshot = strategy.snapshot()

        tokens_sent_est = count_tokens_approximately(_to_lc_messages(messages_to_send))
        messages_sent = len(messages_to_send)

        t0 = time.perf_counter()
        invoke_result = agent.invoke({"messages": messages_to_send})
        latency_s = time.perf_counter() - t0
        ai_msg, assistant_text = _latest_assistant_message(invoke_result["messages"])
        strategy.add_agent_response(invoke_result["messages"])

        usage = _extract_usage_tokens(ai_msg)
        estimated_output_tokens = count_tokens_approximately([AIMessage(content=assistant_text)])
        input_tokens = usage["input_tokens"] if usage["input_tokens"] is not None else tokens_sent_est
        output_tokens = (
            usage["output_tokens"] if usage["output_tokens"] is not None else estimated_output_tokens
        )
        total_tokens = (
            usage["total_tokens"]
            if usage["total_tokens"] is not None
            else int(input_tokens) + int(output_tokens)
        )
        estimated_cost_usd = _estimate_turn_cost(int(input_tokens), int(output_tokens), args)

        report.total_latency_s += latency_s
        report.total_invoke_calls += 1
        report.total_tokens_sent_est += tokens_sent_est
        report.total_input_tokens += int(input_tokens)
        report.total_output_tokens += int(output_tokens)
        report.total_tokens += int(total_tokens)
        report.total_estimated_cost_usd += estimated_cost_usd

        turn_record: dict[str, Any] = {
            "run_id": run_id,
            "replicate_idx": replicate_idx,
            "scenario": scenario.name,
            "competency": scenario.competency,
            "strategy": strategy_slug,
            "strategy_name": strategy.name,
            "turn_idx": turn_idx,
            "turn_kind": turn.kind.value,
            "user_message": turn.user_message,
            "assistant_response": assistant_text,
            "messages_sent_count": messages_sent,
            "tokens_sent_est": tokens_sent_est,
            "latency_s": round(latency_s, 6),
            "memory_snapshot": memory_snapshot,
            "messages_sent_preview": _serialize_messages_for_trace(messages_to_send),
            "probe_label": turn.probe_label,
            "expected": turn.expected,
            "judge_result": None,
            "judge_rationale": None,
            "deterministic_match": None,
            "final_adjudicated_result": None,
            "adjudication_method": None,
            "adjudication_note": None,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "estimated_cost_usd": round(estimated_cost_usd, 8),
        }

        if turn.is_probe():
            if turn.expected is None:
                raise ValueError(f"Probe turn {turn.probe_label} is missing expected text.")
            probe_label = turn.probe_label or f"probe_{turn_idx}"
            (
                raw_judge_result,
                raw_judge_rationale,
                deterministic,
                final_result,
                adjudication_method,
                adjudication_note,
            ) = _adjudicate_probe(
                scenario_name=scenario.name,
                strategy_slug=strategy_slug,
                probe_label=probe_label,
                response=assistant_text,
                expected=turn.expected,
                expected_keywords=turn.expected_keywords,
                judge_model_str=args.judge_model,
                overrides=overrides,
            )
            expected_keywords = _extract_keywords(turn.expected, turn.expected_keywords)
            probe_result = ProbeResult(
                run_id=run_id,
                replicate_idx=replicate_idx,
                scenario=scenario.name,
                competency=scenario.competency,
                strategy=strategy_slug,
                strategy_name=strategy.name,
                label=probe_label,
                expected=turn.expected,
                expected_keywords=expected_keywords,
                response=assistant_text,
                raw_judge_result=raw_judge_result,
                raw_judge_rationale=raw_judge_rationale,
                deterministic_match=deterministic,
                final_adjudicated_result=final_result,
                adjudication_method=adjudication_method,
                adjudication_note=adjudication_note,
                latency_s=latency_s,
                messages_sent=messages_sent,
                tokens_sent_est=tokens_sent_est,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(total_tokens),
                estimated_cost_usd=estimated_cost_usd,
            )
            report.probe_results.append(probe_result)
            writer.write_probe_result(probe_result.__dict__)

            turn_record["judge_result"] = raw_judge_result
            turn_record["judge_rationale"] = raw_judge_rationale
            turn_record["deterministic_match"] = deterministic
            turn_record["final_adjudicated_result"] = final_result
            turn_record["adjudication_method"] = adjudication_method
            turn_record["adjudication_note"] = adjudication_note

            if args.verbose:
                status = "PASS" if final_result else "FAIL"
                print(
                    f"[{turn_idx:02d}] PROBE {probe_label}: {status} "
                    f"(raw={raw_judge_result}, deterministic={deterministic}, method={adjudication_method})"
                )
        elif args.verbose:
            print(
                f"[{turn_idx:02d}] {turn.kind.value.upper()}: "
                f"{latency_s:.2f}s, sent_est={tokens_sent_est}, total={total_tokens}"
            )

        writer.write_turn_trace(turn_record)

    return report


def _mean_ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0.0}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1.0}
    std = statistics.stdev(values)
    margin = 1.96 * (std / math.sqrt(len(values)))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "n": float(len(values)),
    }


def _aggregate_trials(trials: list[TrialReport]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[TrialReport]] = {}
    for trial in trials:
        key = (trial.scenario_name, trial.strategy_slug)
        grouped.setdefault(key, []).append(trial)

    rows: list[dict[str, Any]] = []
    for (scenario_name, strategy_slug), group in sorted(grouped.items()):
        sample = group[0]
        raw_recall_values = [
            (trial.raw_pass_count / trial.probe_count * 100.0) if trial.probe_count else 0.0
            for trial in group
        ]
        final_recall_values = [
            (trial.final_pass_count / trial.probe_count * 100.0) if trial.probe_count else 0.0
            for trial in group
        ]
        avg_latency_values = [
            (trial.total_latency_s / trial.total_invoke_calls) if trial.total_invoke_calls else 0.0
            for trial in group
        ]
        avg_tokens_values = [
            (trial.total_tokens / trial.total_invoke_calls) if trial.total_invoke_calls else 0.0
            for trial in group
        ]
        avg_cost_values = [
            (trial.total_estimated_cost_usd / trial.total_invoke_calls) if trial.total_invoke_calls else 0.0
            for trial in group
        ]
        disagreement_rate_values = [
            (trial.disagreement_count / trial.probe_count * 100.0) if trial.probe_count else 0.0
            for trial in group
        ]

        raw_ci = _mean_ci95(raw_recall_values)
        final_ci = _mean_ci95(final_recall_values)
        latency_ci = _mean_ci95(avg_latency_values)
        tokens_ci = _mean_ci95(avg_tokens_values)
        cost_ci = _mean_ci95(avg_cost_values)
        disagreement_ci = _mean_ci95(disagreement_rate_values)

        row = {
            "scenario": scenario_name,
            "competency": sample.scenario_competency,
            "failure_interpretation": sample.failure_interpretation,
            "strategy": strategy_slug,
            "strategy_name": sample.strategy_name,
            "strategy_desc": sample.strategy_desc,
            "replicates": len(group),
            "probe_count_per_replicate": sample.probe_count,
            "turn_count_per_replicate": sample.total_invoke_calls,
            "session_breaks_per_replicate": sample.session_break_count,
            "raw_recall_mean_pct": raw_ci["mean"],
            "raw_recall_ci95_low_pct": raw_ci["ci95_low"],
            "raw_recall_ci95_high_pct": raw_ci["ci95_high"],
            "final_recall_mean_pct": final_ci["mean"],
            "final_recall_ci95_low_pct": final_ci["ci95_low"],
            "final_recall_ci95_high_pct": final_ci["ci95_high"],
            "avg_latency_mean_s": latency_ci["mean"],
            "avg_latency_ci95_low_s": latency_ci["ci95_low"],
            "avg_latency_ci95_high_s": latency_ci["ci95_high"],
            "avg_total_tokens_mean": tokens_ci["mean"],
            "avg_total_tokens_ci95_low": tokens_ci["ci95_low"],
            "avg_total_tokens_ci95_high": tokens_ci["ci95_high"],
            "avg_cost_mean_usd": cost_ci["mean"],
            "avg_cost_ci95_low_usd": cost_ci["ci95_low"],
            "avg_cost_ci95_high_usd": cost_ci["ci95_high"],
            "adjudication_disagreement_rate_mean_pct": disagreement_ci["mean"],
            "adjudication_disagreement_rate_ci95_low_pct": disagreement_ci["ci95_low"],
            "adjudication_disagreement_rate_ci95_high_pct": disagreement_ci["ci95_high"],
        }
        rows.append(row)
    return rows


def print_report(aggregated_rows: list[dict[str, Any]]) -> None:
    """Print aggregated report grouped by scenario and strategy."""
    print("\n" + "=" * 116)
    print(" MEMORY STRATEGY BENCHMARK REPORT (MEAN +/- 95% CI)")
    print("=" * 116)
    print(
        f"\n{'Scenario':<30} | {'Strategy':<16} | {'Final Recall':>24} | "
        f"{'Avg Latency(s)':>16} | {'Avg Tokens':>12} | {'Avg Cost($)':>12}"
    )
    print("-" * 116)
    for row in aggregated_rows:
        final_text = (
            f"{row['final_recall_mean_pct']:.1f}% "
            f"[{row['final_recall_ci95_low_pct']:.1f}, {row['final_recall_ci95_high_pct']:.1f}]"
        )
        latency_text = (
            f"{row['avg_latency_mean_s']:.2f} "
            f"[{row['avg_latency_ci95_low_s']:.2f}, {row['avg_latency_ci95_high_s']:.2f}]"
        )
        token_text = (
            f"{row['avg_total_tokens_mean']:.0f} "
            f"[{row['avg_total_tokens_ci95_low']:.0f}, {row['avg_total_tokens_ci95_high']:.0f}]"
        )
        cost_text = (
            f"{row['avg_cost_mean_usd']:.5f} "
            f"[{row['avg_cost_ci95_low_usd']:.5f}, {row['avg_cost_ci95_high_usd']:.5f}]"
        )
        print(
            f"{row['scenario']:<30} | {row['strategy']:<16} | {final_text:>24} | "
            f"{latency_text:>16} | {token_text:>12} | {cost_text:>12}"
        )
    print()


def _build_run_summary(
    *,
    run_id: str,
    args,
    trials: list[TrialReport],
    aggregated_rows: list[dict[str, Any]],
    writer: RunArtifactWriter,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "judge_model": args.judge_model,
        "selected_scenarios": args.scenario,
        "selected_strategy": args.strategy,
        "replicates": args.replicates,
        "pricing": {
            "input_cost_per_1k": args.input_cost_per_1k,
            "output_cost_per_1k": args.output_cost_per_1k,
        },
        "output_paths": writer.output_paths,
        "results": aggregated_rows,
        "trial_count": len(trials),
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Memory strategy benchmark harness")
    parser.add_argument(
        "--strategy",
        choices=[*STRATEGY_NAMES, "all"],
        default="all",
        help="Memory strategy to run (default: all)",
    )
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="recency_vs_distance",
        help="Scenario to run (default: recency_vs_distance)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Independent repeats per scenario x strategy to compute CI (default: 1)",
    )
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string used for agent responses",
    )
    parser.add_argument(
        "--judge-model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string used for LLM-as-judge scoring",
    )
    parser.add_argument(
        "--window-k",
        type=int,
        default=6,
        help="Exchanges kept by window/profile/hybrid memories",
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=4000,
        help="Token threshold for summary/hybrid memory summarization",
    )
    parser.add_argument(
        "--summary-keep-recent",
        type=int,
        default=4,
        help="Recent exchanges kept verbatim by summary/hybrid memory",
    )
    parser.add_argument(
        "--longterm-store-path",
        default="evals/longterm_memory_store.json",
        help="Storage file used by longterm_profile strategy",
    )
    parser.add_argument(
        "--memory-key",
        default="demo_user",
        help="Base memory key used by longterm_profile strategy",
    )
    parser.add_argument(
        "--output-dir",
        default="evals/runs",
        help="Directory to store run artifacts",
    )
    parser.add_argument(
        "--override-file",
        default=None,
        help="Optional JSON file with manual probe adjudication overrides",
    )
    parser.add_argument(
        "--input-cost-per-1k",
        type=float,
        default=0.0,
        help="Cost proxy: USD per 1K input tokens",
    )
    parser.add_argument(
        "--output-cost-per-1k",
        type=float,
        default=0.0,
        help="Cost proxy: USD per 1K output tokens",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-turn details",
    )
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")

    selected_scenarios = (
        list(SCENARIOS.values()) if args.scenario == "all" else [get_scenario(args.scenario)]
    )
    selected_strategies = list(STRATEGY_NAMES) if args.strategy == "all" else [args.strategy]

    run_id = generate_run_id()
    writer = RunArtifactWriter(args.output_dir, run_id)
    overrides = _load_overrides(args.override_file)

    print(f"Run ID: {run_id}")
    print(
        f"Running {args.replicates} replicate(s) x {len(selected_scenarios)} scenario(s) x "
        f"{len(selected_strategies)} strategy(ies)"
    )

    trials: list[TrialReport] = []
    for replicate_idx in range(args.replicates):
        print(f"\n=== Replicate {replicate_idx + 1}/{args.replicates} ===")
        for scenario in selected_scenarios:
            for strategy_slug in selected_strategies:
                print(f">> Scenario={scenario.name} | Strategy={strategy_slug}")
                trial = run_trial(
                    run_id=run_id,
                    replicate_idx=replicate_idx,
                    scenario=scenario,
                    strategy_slug=strategy_slug,
                    args=args,
                    writer=writer,
                    overrides=overrides,
                )
                trials.append(trial)
                print(
                    f"   Done: final recall {trial.final_pass_count}/{trial.probe_count} "
                    f"({trial.total_latency_s:.1f}s, cost=${trial.total_estimated_cost_usd:.5f})"
                )

    aggregated_rows = _aggregate_trials(trials)
    summary = _build_run_summary(
        run_id=run_id,
        args=args,
        trials=trials,
        aggregated_rows=aggregated_rows,
        writer=writer,
    )
    writer.write_summary(summary)

    print_report(aggregated_rows)
    print(f"Artifacts written to: {writer.run_dir}")
    print(f" - {writer.summary_path.name}")
    print(f" - {writer.turn_traces_path.name}")
    print(f" - {writer.probe_results_path.name}")


if __name__ == "__main__":
    main()
