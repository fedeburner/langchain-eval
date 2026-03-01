# Memory Strategy Benchmark

This started as a barebones CLI chat agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). I took it and built out six different memory strategies, an evaluation harness to systematically test them, and a Streamlit dashboard to dig into the results.

The question I wanted to answer: **how do different memory approaches actually affect an agent’s ability to recall facts, deal with contradictions, and hold context over long conversations?**

## Memory Strategies

I went with six strategies that cover a pretty wide range of the design space:

**Full Buffer** just keeps everything. Every message stays in the context window forever. Best possible recall, but token count grows without bound. Works fine for short chats, gets expensive fast for long ones.

**Sliding Window** uses LangChain’s `trim_messages` to keep only the last K exchanges (I defaulted to `6`). Token usage stays flat no matter how long the conversation runs. The obvious downside is that anything older than K turns is permanently gone.

**Running Summary** watches the token count and, once it crosses a threshold, asks the LLM to compress older messages into a paragraph. After that it sends the summary plus recent turns. This preserves the general shape of the conversation without the linear cost growth, but specific details can get lost in the compression.

**Structured Profile** scans each user message with regex patterns looking for facts like name, location, profession, and pet info. It stores these in a dictionary and injects them as a system message each turn, alongside a recent-turn window. The upside is that it’s very interpretable (you can literally see what it “knows”). The downside is it only catches facts that match its patterns.

**Hybrid** layers all three together: profile extraction, running summary, and recent window. In theory this should be the best of everything. In practice it mostly is, but it also inherits each component’s weaknesses. The Profile regex bug affects Hybrid too, for example.

**Long-Term Profile** is the same as Profile but it writes the extracted facts to a JSON file on disk. When a session ends and a new one starts, it loads the previous profile back in. This is the only strategy that remembers anything across session boundaries.

---

## Evaluation Approach

### Scenarios

Instead of one big generic conversation, I wrote eight scenarios that each test a specific thing. The idea is that when something fails, the scenario name tells you what kind of memory problem caused it.

| Scenario | What it tests | Turns |
|----------|---------------|-------|
| Long-Range Recall | Do facts from early turns survive after lots of filler? | 11 |
| Conflict Resolution | When the user corrects a fact, does the agent use the new value? | 7 |
| Compositional Recall | Can it combine facts from separate, non-adjacent turns? | 9 |
| Noise Robustness | Do important facts survive heavy irrelevant chatter? | 10 |
| Style Preference | Does it remember response-format instructions? | 5 |
| Cross-Session Memory | After a session reset, is anything retained? | 9 |
| Deep Conversation | After 20+ filler turns, can it recall details from turn 2? | 26 |
| Repeated Updates | When the same fact changes 3 times, does it track the latest? | 10 |

Each scenario has three turn types: **fact** turns where the user states something memorable, **filler** turns with unrelated questions that push facts further back in history, and **probe** turns that ask the agent to recall specific facts.

## MODEL USED

Everything runs on Claude Haiku. I chose it because it’s fast and cheap enough to run all `48` trials, which is about `6M` tokens (it ended up being between `$6` and `$14`). It also helps to use a “dumber” model here so the memory strategy behaviors show up more clearly.

### Scoring

Every probe gets scored two ways:

1. An **LLM judge** (Claude Haiku) reads the agent’s answer and the expected fact and decides if they match.
2. A **keyword matcher** checks for specific expected strings in the response.

If they disagreed, the keyword match wins. This turned out to be important: the LLM judge gave false negatives on about `8%` of probes, and the keyword layer caught every one of them.

NOTE: In hindsight, I probably should’ve used a different model than Haiku for the LLM judge. It’s not ideal to evaluate a model with itself, and it clearly misjudged some results (you can see this in the Streamlit/Loom). But the two-phase approach with the keyword match was able to correct for these.

---

## SUMMARY OF RESULTS

Six strategies, eight scenarios, `48` total trials. `522` conversation turns processed, about `6M` tokens consumed. `72` out of `114` probes passed — a `63%` overall rate. The hardest scenario was Cross-Session Memory at `11%` average pass rate, and the easiest was Style Preference at `100%`.

## DEEPER / FUN FINDINGS

**Hybrid and Summary have complementary blind spots.** This was the most surprising result. I assumed Hybrid (Profile + Summary + Window combined) would dominate since it theoretically gets the best of each piece. Instead, Hybrid won on Deep Conversation (`100%` vs Summary’s `0%`) because its Profile extractor caught things like “wife’s name is Sarah” that the summarizer compressed away. But then Summary won on Noise Robustness (`100%` vs Hybrid’s `0%`) because LLM compression kept “emergency contact is Alice” alive — a fact that didn’t match any of Profile’s regex patterns. So combining strategies can actually hurt if one component drops facts the other would have kept.

**The Profile extractor has a real bug.** In every single scenario, the name field stored “Jordan And I Am A Data Engineer At Stripe” instead of just “Jordan.” The regex `my name is ([a-z][a-z' -]{1,40})` greedily matches past the name into the rest of the sentence. I left this unfixed intentionally — fixing this one regex is trivial, but there will always be another pattern you didn’t anticipate (“emergency contact is Alice,” “my wife’s name is Sarah”). The bug demonstrates the fundamental limitation of regex-based extraction and makes the case for LLM-based fact extraction in production.

**Full Buffer’s token usage blows up.** It scored the highest recall (`88%`), but in the Deep Conversation scenario (`26` turns), tokens went from `6,222` to `111,578`. That’s an `18x` increase over one conversation. For a platform that runs for hours a day, that’s not viable.

**Summary barely ever summarizes.** The summarization logic only kicked in on `1` of the `8` scenarios. The rest were short enough that the token threshold was never hit, so Summary basically acted like Full Buffer. I could’ve lowered the threshold to force more summarization, but that would be tuning the strategy to fit the benchmark rather than testing the strategy as designed. The finding itself is useful: in real usage, most conversations are short enough that Summary provides no benefit over Full Buffer — its cost savings only matter for genuinely long sessions.

**The evaluator disagrees with itself.** I scored each probe two ways — an LLM judge and a keyword matcher. They disagreed on `~8%` of probes. Every time, the LLM judge gave a false negative (said the answer was wrong when it was actually right). The keyword fallback caught all of these. It’s a good reminder that LLM-based eval needs a sanity-check layer.

---

## Getting Started

You need Python `3.13+`, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

```bash
git clone https://github.com/fedeburner/langchain-eval.git
cd langchain-eval
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
````

### Trying the CLI

The quickest way to poke at this is the chat REPL. You pick a memory strategy with `--memory`:

```bash
uv run chat                           # defaults to full buffer
uv run chat --memory window           # sliding window, last 6 exchanges
uv run chat --memory summary          # LLM-compressed summary + recent turns
uv run chat --memory profile          # regex-based fact extraction + recent turns
uv run chat --memory hybrid           # profile + summary + window combined
uv run chat --memory longterm_profile # persists profile to disk across sessions
```

Type `quit` or `exit` to end. You can also swap models with `--model openai:gpt-4o` or `--model google_genai:gemini-2.5-flash`.

### Running the benchmark

The harness runs scripted conversations and checks whether the agent remembered what it was supposed to:

```bash
# Full benchmark: 8 scenarios x 6 strategies
uv run python -m evals.harness --scenario all --strategy all --replicates 1 --output-dir evals/runs

# Or just one combo
uv run python -m evals.harness --strategy hybrid --scenario recency_vs_distance
```

Each run writes three files to `evals/runs/<run_id>/`:

* `run_summary.json` -- recall, latency, and token numbers per scenario x strategy
* `turn_traces.jsonl` -- every turn with the full conversation, memory snapshot, and probe scoring
* `probe_results.jsonl` -- per-probe detail: LLM judge verdict, keyword match, final result

There’s a pre-computed run already in the repo, so the dashboard works right away without needing to run anything.

### Viewing results

```bash
uv run streamlit run app/dashboard.py
```

The dashboard is one scrollable page. At the top there’s an overview of the full benchmark scope (strategies, scenarios, total turns, tokens consumed), a strategy ranking table, and five key findings. Then there’s a color-coded scorecard grid — green for `100%` recall, yellow for partial, red for `0%`. Click any cell and it shows you the actual conversation with a verdict panel next to it: what the agent was asked, what it said, whether it passed, why it failed, and what was in memory when the probe happened. At the bottom there’s a table comparing how all six strategies handled the same probes.

---

## Running Tests

```bash
uv run pytest evals/ -v
```

---

## Project Structure

```text
take-home/
├── pyproject.toml
├── README.md
├── app/
│   ├── dashboard.py           # Streamlit app shell
│   ├── dashboard_data.py      # Data loading + scenario labels
│   └── dashboard_views.py     # All rendering (scorecard, verdict, chat, comparison)
├── src/
│   └── agent/
│       ├── core.py            # Agent factory (wraps LangChain init_chat_model)
│       ├── cli.py             # Chat REPL with --memory flag
│       └── memory.py          # All six memory strategy implementations
└── evals/
    ├── test_agent.py          # Smoke tests
    ├── scenarios.py           # Eight benchmark scenarios
    ├── reporting.py           # Artifact writing helpers
    └── harness.py             # Evaluation harness
```

```
```
