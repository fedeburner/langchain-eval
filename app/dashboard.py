"""Clean memory strategy benchmark dashboard."""

from __future__ import annotations

import streamlit as st

import dashboard_data as data
import dashboard_views as views


def main() -> None:
    st.set_page_config(
        page_title="Memory Benchmark",
        layout="wide",
        menu_items={},
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        "<style>"
        "[data-testid='stMainMenu'], [data-testid='stToolbar'] {display: none !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    runs = data.list_available_runs()
    if not runs:
        st.warning("No runs found in `evals/runs/`. Run `uv run python -m evals.harness` first.")
        return

    selected_run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
    bundle = data.load_run(str(selected_run))
    summary = bundle["summary"]
    results_df = bundle["results"]
    turns_df = bundle["turns"]
    probes_df = bundle["probes"]

    st.sidebar.caption(f"Model: {summary.get('model', 'n/a')}")
    st.sidebar.caption(f"Judge: {summary.get('judge_model', 'n/a')}")

    if results_df.empty:
        st.error("This run has no results.")
        return

    # 1. Overview + findings
    st.markdown("## Memory Strategy Benchmark")
    views.render_overview(results_df, turns_df, probes_df)

    st.divider()

    # 2. Scorecard grid
    selected_scenario, selected_strategy = views.render_scorecard(results_df)

    st.divider()

    # 3. Verdict for selected cell
    views.render_detail(
        selected_scenario, selected_strategy,
        turns_df, probes_df, results_df,
    )

    st.divider()

    # 4. Conversation transcript
    views.render_chat(selected_scenario, selected_strategy, turns_df)

    st.divider()

    # 5. Cross-strategy comparison (with its own scenario picker)
    views.render_comparison(
        selected_scenario, selected_strategy,
        turns_df, probes_df, results_df,
    )


if __name__ == "__main__":
    main()
