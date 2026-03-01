"""View renderers for the memory benchmark dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import dashboard_data as data


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def _recall_icon(pct: float) -> str:
    if pct >= 99.9:
        return "\u2705"
    if pct > 0:
        return "\u26a0\ufe0f"
    return "\u274c"


def render_scorecard(results_df: pd.DataFrame) -> tuple[str, str]:
    if results_df.empty:
        st.warning("No benchmark results found for this run.")
        return "", ""

    scenarios = [s for s in data.SCENARIO_ORDER if s in results_df["scenario"].values]
    for s in sorted(results_df["scenario"].unique()):
        if s not in scenarios:
            scenarios.append(s)

    strategies = sorted(results_df["strategy"].unique().tolist())
    strategy_names: dict[str, str] = {}
    for _, row in results_df.drop_duplicates("strategy").iterrows():
        strategy_names[row["strategy"]] = data.strategy_label(row.to_dict())

    pivot = results_df.pivot_table(
        index="scenario", columns="strategy", values="final_recall_mean_pct", aggfunc="mean"
    ).reindex(index=scenarios, columns=strategies)

    if "selected_scenario" not in st.session_state:
        st.session_state["selected_scenario"] = scenarios[0] if scenarios else ""
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = strategies[0] if strategies else ""

    col_widths = [2] + [1] * len(strategies) + [1]
    header_cols = st.columns(col_widths)
    header_cols[0].markdown("**Scenario**")
    for i, strat in enumerate(strategies):
        header_cols[i + 1].markdown(f"**{strategy_names.get(strat, strat)}**")
    header_cols[-1].markdown("**Avg**")

    for scenario in scenarios:
        row_cols = st.columns(col_widths)
        subtitle = data.scenario_subtitle(scenario)
        if subtitle:
            row_cols[0].markdown(f"**{data.scenario_label(scenario)}**  \n*{subtitle}*")
        else:
            row_cols[0].markdown(f"**{data.scenario_label(scenario)}**")
        row_values: list[float] = []
        for i, strat in enumerate(strategies):
            pct = pivot.loc[scenario, strat] if scenario in pivot.index and strat in pivot.columns else None
            if pct is None or pd.isna(pct):
                row_cols[i + 1].write("--")
                continue
            row_values.append(float(pct))
            icon = _recall_icon(pct)
            is_selected = (
                st.session_state["selected_scenario"] == scenario
                and st.session_state["selected_strategy"] == strat
            )
            label = f"{icon} {pct:.0f}%"
            if is_selected:
                label = f"**[ {icon} {pct:.0f}% ]**"
            if row_cols[i + 1].button(label, key=f"sc_{scenario}_{strat}", use_container_width=True):
                st.session_state["selected_scenario"] = scenario
                st.session_state["selected_strategy"] = strat
                st.rerun()
        row_mean = sum(row_values) / len(row_values) if row_values else 0.0
        row_cols[-1].markdown(f"**{row_mean:.0f}%**")

    footer_cols = st.columns(col_widths)
    footer_cols[0].markdown("**Avg**")
    all_means: list[float] = []
    for i, strat in enumerate(strategies):
        col_vals = [float(pivot.loc[sc, strat]) for sc in scenarios
                     if sc in pivot.index and strat in pivot.columns and not pd.isna(pivot.loc[sc, strat])]
        col_mean = sum(col_vals) / len(col_vals) if col_vals else 0.0
        all_means.append(col_mean)
        footer_cols[i + 1].markdown(f"**{col_mean:.0f}%**")
    grand_mean = sum(all_means) / len(all_means) if all_means else 0.0
    footer_cols[-1].markdown(f"**{grand_mean:.0f}%**")

    by_strategy = (
        results_df.groupby("strategy", as_index=False)
        .agg(
            recall=("final_recall_mean_pct", "mean"),
            tokens=("avg_total_tokens_mean", "mean"),
            latency=("avg_latency_mean_s", "mean"),
        )
        .sort_values("recall", ascending=False)
    )
    if not by_strategy.empty:
        best = by_strategy.iloc[0]
        worst = by_strategy.iloc[-1]
        best_name = strategy_names.get(best["strategy"], best["strategy"])
        worst_name = strategy_names.get(worst["strategy"], worst["strategy"])
        st.caption(
            f"Best overall: **{best_name}** at {best['recall']:.0f}% avg recall. "
            f"Weakest: **{worst_name}** at {worst['recall']:.0f}%."
        )

    return (
        str(st.session_state.get("selected_scenario", "")),
        str(st.session_state.get("selected_strategy", "")),
    )


# ---------------------------------------------------------------------------
# Overview + findings
# ---------------------------------------------------------------------------

def render_overview(
    results_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    probes_df: pd.DataFrame,
) -> None:
    if results_df.empty or turns_df.empty or probes_df.empty:
        return

    total_strategies = turns_df["strategy"].nunique()
    total_scenarios = turns_df["scenario"].nunique()
    total_trials = turns_df.groupby(["scenario", "strategy"]).ngroups
    total_turns = len(turns_df)
    total_probes = len(probes_df)
    probes_passed = int(probes_df["final_adjudicated_result"].sum())
    total_tokens = int(turns_df["total_tokens"].sum())

    # Top-level overview strip
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("Strategies", str(total_strategies))
    o2.metric("Scenarios", str(total_scenarios))
    o3.metric("Trials", str(total_trials))
    o4.metric("Turns", f"{total_turns:,}")
    o5.metric("Probes", f"{probes_passed}/{total_probes} passed")
    o6.metric("Tokens Used", f"{total_tokens:,.0f}")

    # Strategy ranking table
    by_strategy = (
        results_df.groupby("strategy", as_index=False)
        .agg(recall=("final_recall_mean_pct", "mean"))
        .sort_values("recall", ascending=False)
    )
    strat_tokens = turns_df.groupby("strategy")["total_tokens"].sum().to_dict()
    strat_latency = turns_df.groupby("strategy")["latency_s"].sum().to_dict()
    strat_probes = probes_df.groupby("strategy").agg(
        passed=("final_adjudicated_result", "sum"),
        total=("final_adjudicated_result", "count"),
    ).to_dict("index")

    ranking = by_strategy.copy()
    ranking["tokens_total"] = ranking["strategy"].map(strat_tokens).fillna(0).astype(int)
    ranking["latency_total"] = ranking["strategy"].map(strat_latency).fillna(0).round(0).astype(int)
    ranking["probes"] = ranking["strategy"].map(
        lambda s: f"{int(strat_probes.get(s, {}).get('passed', 0))}/{int(strat_probes.get(s, {}).get('total', 0))}"
    )
    ranking["recall"] = ranking["recall"].round(0).astype(int)
    ranking = ranking.rename(columns={
        "strategy": "Strategy",
        "recall": "Recall (%)",
        "tokens_total": "Total Tokens",
        "latency_total": "Total Latency (s)",
        "probes": "Probes Passed",
    })
    st.dataframe(ranking, width="stretch", hide_index=True)

    # Scenario difficulty
    sc_pass = probes_df.groupby("scenario")["final_adjudicated_result"].mean().sort_values()
    hardest = sc_pass.index[0] if not sc_pass.empty else ""
    easiest = sc_pass.index[-1] if not sc_pass.empty else ""
    st.caption(
        f"Hardest scenario: **{data.scenario_label(hardest)}** "
        f"({sc_pass.iloc[0]*100:.0f}% avg pass). "
        f"Easiest: **{data.scenario_label(easiest)}** "
        f"({sc_pass.iloc[-1]*100:.0f}% avg pass)."
    )

    # Key findings
    st.markdown("##### Key Findings")

    max_tokens_val = 0
    max_tokens_first = 0
    for (sc, strat), g in turns_df.groupby(["scenario", "strategy"]):
        if g.empty:
            continue
        last = int(g.iloc[-1].get("total_tokens", 0))
        first = int(g.iloc[0].get("total_tokens", 0))
        if last > max_tokens_val:
            max_tokens_val = last
            max_tokens_first = first

    sum_turns = turns_df[turns_df["strategy"] == "summary"]
    total_sc = sum_turns["scenario"].nunique()
    summarized_sc = 0
    for sc, g in sum_turns.groupby("scenario"):
        snaps = [s for s in g["memory_snapshot"] if isinstance(s, dict)]
        if any(s.get("has_summary") for s in snaps):
            summarized_sc += 1

    disagree = probes_df[probes_df["raw_judge_result"] != probes_df["final_adjudicated_result"]]
    disagree_pct = len(disagree) / len(probes_df) * 100 if len(probes_df) else 0

    name_bugs = set()
    for _, r in turns_df.iterrows():
        snap = r.get("memory_snapshot")
        if isinstance(snap, dict):
            name = snap.get("profile", {}).get("name", "")
            if name and len(name) > 20:
                name_bugs.add(name)

    inversions = 0
    for sc in probes_df["scenario"].unique():
        h = probes_df[(probes_df["scenario"]==sc)&(probes_df["strategy"]=="hybrid")]
        s = probes_df[(probes_df["scenario"]==sc)&(probes_df["strategy"]=="summary")]
        if h.empty or s.empty:
            continue
        h_pct = h["final_adjudicated_result"].mean() * 100
        s_pct = s["final_adjudicated_result"].mean() * 100
        if abs(h_pct - s_pct) > 10:
            inversions += 1

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Peak Token Growth", f"{max_tokens_first:,} \u2192 {max_tokens_val:,}")
    c1.caption("Full Buffer grows unboundedly")
    c2.metric("Summary Triggered", f"{summarized_sc}/{total_sc} scenarios")
    c2.caption("Too short to summarize in most cases")
    c3.metric("Judge Disagreements", f"{len(disagree)}/{len(probes_df)} ({disagree_pct:.0f}%)")
    c3.caption("Two-layer eval catches false negatives")
    c4.metric("Strategy Inversions", f"{inversions} scenarios")
    c4.caption("Hybrid & Summary beat each other")
    c5.metric("Regex Bugs", f"{len(name_bugs)} patterns")
    c5.caption("Profile captures too much text")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_display_name(label: str) -> str:
    replacements = {
        "rvd_dog_recall": "Dog name & breed",
        "rvd_location_recall": "Location",
        "rvd_profession_recall": "Profession",
        "conflict_current_location": "Current location",
        "conflict_old_location": "Previous location",
        "compose_role_languages": "Role & languages",
        "compose_identity_and_role": "Name & team",
        "noise_emergency_contact": "Emergency contact",
        "noise_blood_type": "Blood type",
        "pref_style_recall": "Style preference recall",
        "pref_style_application": "Style preference applied",
        "longterm_name_after_break": "Name (after break)",
        "longterm_location_after_break": "Location (after break)",
        "longterm_updated_location": "Updated location (after break)",
        "deep_spouse_recall": "Spouse name & job",
        "deep_cats_recall": "Pet names",
        "deep_job_recall": "Job & company",
        "update_current_city": "Current city",
        "update_city_history": "City history",
    }
    return replacements.get(label, label.replace("_", " ").title())


def _explain_failure(snapshot: dict[str, Any], conv: pd.DataFrame, scenario: str) -> str:
    reasons: list[str] = []
    strategy = str(snapshot.get("strategy", ""))
    total_turns = len(conv) if not conv.empty else 0

    if "window_exchanges" in snapshot:
        window_k = snapshot["window_exchanges"]
        if total_turns > window_k * 2:
            reasons.append(
                f"Window keeps only last {window_k} exchanges but conversation had {total_turns} turns."
            )

    profile = snapshot.get("profile")
    if isinstance(profile, dict):
        missing = [k for k, v in profile.items()
                   if k not in ("updates", "preferences", "learning_languages") and not v]
        if missing:
            reasons.append(f"Profile extraction missed: {', '.join(missing)}.")

    if scenario == "cross_session_longterm" and strategy != "longterm_profile":
        reasons.append("No persistence layer -- memory lost on session reset.")

    if not reasons:
        stored = snapshot.get("stored_message_count", 0)
        if stored < total_turns:
            reasons.append(f"Only {stored}/{total_turns} messages in memory at probe time.")

    return " ".join(reasons) if reasons else ""


# ---------------------------------------------------------------------------
# Selected cell detail (conversation left, verdict right)
# ---------------------------------------------------------------------------

def render_detail(
    scenario: str, strategy: str,
    turns_df: pd.DataFrame, probes_df: pd.DataFrame, results_df: pd.DataFrame,
) -> None:
    """Verdict panel only -- no conversation transcript."""
    if not scenario or not strategy:
        st.info("Click a cell in the scorecard to inspect a conversation.")
        return

    result_row = results_df[
        (results_df["scenario"] == scenario) & (results_df["strategy"] == strategy)
    ]
    if result_row.empty:
        return

    r = result_row.iloc[0]
    strategy_name = data.strategy_label(r.to_dict())
    recall_pct = float(r["final_recall_mean_pct"])
    desc = str(r.get("strategy_desc", ""))

    st.markdown(f"### {data.scenario_label(scenario)}  /  {strategy_name}")
    if desc:
        st.caption(desc)

    conv = data.get_conversation(turns_df, scenario, strategy)
    probes = data.get_probes_for(probes_df, scenario, strategy)

    if conv.empty:
        st.warning("No conversation data found.")
        return

    # Metrics row
    first_tokens = int(conv.iloc[0].get("total_tokens", 0))
    last_tokens = int(conv.iloc[-1].get("total_tokens", 0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall", f"{recall_pct:.0f}%")
    m2.metric("Avg Latency", f"{r.get('avg_latency_mean_s', 0):.1f}s")
    m3.metric("Avg Tokens", f"{r.get('avg_total_tokens_mean', 0):,.0f}")
    if first_tokens > 0 and last_tokens > first_tokens * 2:
        m4.metric("Token Growth", f"{last_tokens/first_tokens:.0f}x")
    else:
        m4.metric("Total Turns", str(len(conv)))

    # Probe results with Expected vs Got
    if not probes.empty:
        for _, p in probes.iterrows():
            passed = bool(p.get("final_adjudicated_result", False))
            icon = "\u2705" if passed else "\u274c"
            label = _probe_display_name(str(p.get("label", "")))
            expected = str(p.get("expected", ""))

            st.markdown(f"{icon} **{label}**")
            exp_col, got_col = st.columns(2)
            exp_col.caption(f"Expected: {expected}")

            probe_turn = conv[conv["probe_label"] == p.get("label")] if "probe_label" in conv.columns else pd.DataFrame()
            if not probe_turn.empty:
                actual = str(probe_turn.iloc[0].get("assistant_response", ""))
                preview = actual[:150] + ("..." if len(actual) > 150 else "")
                got_col.caption(f"Got: {preview}")

            if p.get("raw_judge_result") != p.get("final_adjudicated_result"):
                st.caption(":orange[Judge disagreed -- keyword match overruled]")

    # Why it failed
    probe_turns = conv[conv["turn_kind"] == "probe"]
    if not probe_turns.empty and recall_pct < 99.9:
        last_probe = probe_turns.iloc[-1]
        snapshot = last_probe.get("memory_snapshot")
        if isinstance(snapshot, dict) and snapshot:
            explanation = _explain_failure(snapshot, conv, scenario)
            if explanation:
                st.info(f"**Why {recall_pct:.0f}%:** {explanation}")

    # Memory state
    if not probe_turns.empty:
        snapshot = probe_turns.iloc[-1].get("memory_snapshot")
        if isinstance(snapshot, dict) and snapshot:
            with st.expander("Memory state at probe time"):
                mem_parts: list[str] = []
                mem_parts.append(f"**Messages stored:** {snapshot.get('stored_message_count', '?')}")
                if "window_exchanges" in snapshot:
                    mem_parts.append(f"**Window size:** k={snapshot['window_exchanges']}")
                if snapshot.get("has_summary"):
                    mem_parts.append(f"**Summary:** {snapshot.get('summary_chars', 0)} chars")
                profile = snapshot.get("profile")
                if isinstance(profile, dict):
                    known = [k for k, v in profile.items()
                             if v and v not in ([], None, "")
                             and k not in ("updates", "preferences", "learning_languages")]
                    missing = [k for k, v in profile.items()
                               if k not in ("updates", "preferences", "learning_languages") and not v]
                    if known:
                        mem_parts.append("**Profile extracted:** " + ", ".join(known))
                    if missing:
                        mem_parts.append("**Profile missed:** " + ", ".join(missing))
                st.markdown("  \n".join(mem_parts))


def _render_chat_bubbles(conv: pd.DataFrame) -> None:
    turns_list = list(conv.iterrows())
    i = 0
    while i < len(turns_list):
        _, turn = turns_list[i]
        kind = str(turn.get("turn_kind", "unknown"))
        idx = int(turn.get("turn_idx", 0))

        if kind == "session_break":
            st.divider()
            st.caption(f"Turn {idx} -- :red[session break] -- memory wiped")
            st.divider()
            i += 1
            continue

        if kind == "filler":
            filler_start = i
            while i < len(turns_list) and str(turns_list[i][1].get("turn_kind")) == "filler":
                i += 1
            filler_count = i - filler_start
            first_idx = int(turns_list[filler_start][1].get("turn_idx", 0))
            last_idx = int(turns_list[i - 1][1].get("turn_idx", 0))
            with st.expander(
                f":gray[Turns {first_idx}-{last_idx}: {filler_count} filler "
                f"{'turn' if filler_count == 1 else 'turns'}]",
                expanded=False,
            ):
                for j in range(filler_start, filler_start + filler_count):
                    _, ft = turns_list[j]
                    st.caption(f"Turn {int(ft.get('turn_idx', 0))} :gray[filler]")
                    with st.chat_message("user"):
                        st.write(str(ft.get("user_message", "")))
                    with st.chat_message("assistant"):
                        st.write(str(ft.get("assistant_response", "")))
            continue

        is_probe = kind == "probe"
        badge = ":orange[probe]" if is_probe else ":blue[fact]"
        st.caption(f"Turn {idx} {badge}")

        with st.chat_message("user"):
            st.write(str(turn.get("user_message", "")))
        with st.chat_message("assistant"):
            response = str(turn.get("assistant_response", ""))
            st.write(response if response else "_No response._")

        if is_probe:
            passed = bool(turn.get("final_adjudicated_result", False))
            expected = str(turn.get("expected", ""))
            icon = "\u2705" if passed else "\u274c"
            st.markdown(f"> {icon} **Expected:** {expected}")

        i += 1


# ---------------------------------------------------------------------------
# Standalone chat section
# ---------------------------------------------------------------------------

def render_chat(scenario: str, strategy: str, turns_df: pd.DataFrame) -> None:
    if not scenario or not strategy:
        return
    conv = data.get_conversation(turns_df, scenario, strategy)
    if conv.empty:
        return
    st.markdown(f"#### Conversation Transcript: {data.scenario_label(scenario)} / {strategy.replace('_', ' ').title()}")
    _render_chat_bubbles(conv)

    with st.expander("Raw memory snapshot"):
        last_turn = conv.iloc[-1]
        snapshot = last_turn.get("memory_snapshot")
        if isinstance(snapshot, dict) and snapshot:
            st.json(snapshot)
        else:
            st.info("No snapshot recorded.")


# ---------------------------------------------------------------------------
# Cross-strategy comparison (with scenario navigation)
# ---------------------------------------------------------------------------

def render_comparison(
    selected_scenario: str, selected_strategy: str,
    turns_df: pd.DataFrame, probes_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    if probes_df.empty:
        return

    st.markdown("#### Compare All Strategies on a Scenario")

    scenarios = [s for s in data.SCENARIO_ORDER if s in probes_df["scenario"].values]
    for s in sorted(probes_df["scenario"].unique()):
        if s not in scenarios:
            scenarios.append(s)

    scenario_options = {data.scenario_label(s): s for s in scenarios}
    default_label = data.scenario_label(selected_scenario) if selected_scenario in scenarios else list(scenario_options.keys())[0]

    picked_label = st.selectbox(
        "Pick a scenario to compare all strategies:",
        list(scenario_options.keys()),
        index=list(scenario_options.keys()).index(default_label),
    )
    picked_scenario = scenario_options[picked_label]

    subtitle = data.scenario_subtitle(picked_scenario)
    if subtitle:
        st.caption(subtitle)

    all_probes = probes_df[probes_df["scenario"] == picked_scenario]
    if all_probes.empty:
        st.info("No probe data for this scenario.")
        return

    display_df = all_probes.copy()
    if "label" in display_df.columns:
        display_df["probe"] = display_df["label"].apply(_probe_display_name)
    if "strategy_name" not in display_df.columns and "strategy" in display_df.columns:
        display_df["strategy_name"] = display_df["strategy"]
    if "response" in display_df.columns:
        display_df["agent_said"] = display_df["response"].fillna("").astype(str).str[:120]

    display_df["judge_agreed"] = display_df.apply(
        lambda row: "\u2705" if row.get("raw_judge_result") == row.get("final_adjudicated_result") else "\u26a0\ufe0f",
        axis=1,
    )

    cols = ["strategy_name", "probe", "expected", "agent_said", "final_adjudicated_result", "judge_agreed"]
    present = [c for c in cols if c in display_df.columns]
    if "strategy_name" in display_df.columns:
        display_df = display_df.sort_values(["strategy_name", "probe"])

    col_config = {
        "strategy_name": st.column_config.Column("Strategy"),
        "final_adjudicated_result": st.column_config.Column("Pass/Fail"),
        "agent_said": st.column_config.Column("Agent Said"),
        "judge_agreed": st.column_config.Column("Judge Agreed"),
    }
    st.dataframe(display_df[present], width="stretch", hide_index=True, column_config=col_config)

    disagree_count = len(all_probes[all_probes["raw_judge_result"] != all_probes["final_adjudicated_result"]])
    if disagree_count > 0:
        st.caption(
            f":orange[{disagree_count} judge disagreement(s)] -- "
            f"keyword matching overruled the LLM judge to reduce false negatives."
        )

    sc_results = results_df[results_df["scenario"] == picked_scenario] if not results_df.empty else pd.DataFrame()
    if not sc_results.empty:
        summary_df = sc_results[["strategy_name", "final_recall_mean_pct", "avg_latency_mean_s", "avg_total_tokens_mean"]].copy()
        summary_df = summary_df.rename(columns={
            "strategy_name": "Strategy",
            "final_recall_mean_pct": "Recall (%)",
            "avg_latency_mean_s": "Avg Latency (s)",
            "avg_total_tokens_mean": "Avg Tokens",
        })
        summary_df["Recall (%)"] = summary_df["Recall (%)"].round(0).astype(int)
        summary_df["Avg Latency (s)"] = summary_df["Avg Latency (s)"].round(1)
        summary_df["Avg Tokens"] = summary_df["Avg Tokens"].round(0).astype(int)
        summary_df = summary_df.sort_values("Recall (%)", ascending=False)
        st.caption("Performance on this scenario:")
        st.dataframe(summary_df, width="stretch", hide_index=True)
