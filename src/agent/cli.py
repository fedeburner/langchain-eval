import argparse

from dotenv import load_dotenv

from agent.core import make_agent
from agent.memory import STRATEGY_NAMES, get_strategy, msg_content


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--memory",
        choices=list(STRATEGY_NAMES),
        default="full",
        help=(
            "Memory strategy: full, window, summary, profile, hybrid, longterm_profile. "
            "Use profile/hybrid for user-fact persistence, longterm_profile for cross-session."
        ),
    )
    parser.add_argument(
        "--window-k",
        type=int,
        default=6,
        help="Number of exchanges to keep for window memory (default: 6)",
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=4000,
        help="Token threshold for summary/hybrid memory compression (default: 4000)",
    )
    parser.add_argument(
        "--summary-keep-recent",
        type=int,
        default=4,
        help="Recent exchanges kept verbatim by summary/hybrid memories (default: 4)",
    )
    parser.add_argument(
        "--longterm-store-path",
        default="evals/longterm_memory_store.json",
        help="Storage file used by longterm_profile memory strategy",
    )
    parser.add_argument(
        "--memory-key",
        default="demo_user",
        help="Memory key used by longterm_profile strategy",
    )
    args = parser.parse_args()

    agent = make_agent(args.model, args.system)

    strategy_kwargs = {}
    if args.memory in ("window", "profile", "hybrid", "longterm_profile"):
        strategy_kwargs["k"] = args.window_k
    if args.memory in ("summary", "hybrid"):
        strategy_kwargs["model_str"] = args.model
        strategy_kwargs["max_tokens"] = args.summary_max_tokens
        strategy_kwargs["keep_recent"] = args.summary_keep_recent
    if args.memory == "longterm_profile":
        strategy_kwargs["memory_store_path"] = args.longterm_store_path
        strategy_kwargs["memory_key"] = args.memory_key
    strategy = get_strategy(args.memory, **strategy_kwargs)

    print(f"Chat started (memory: {strategy.name}). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        strategy.add_user_message(user_input)
        messages_to_send = strategy.get_messages()

        result = agent.invoke({"messages": messages_to_send})
        ai_msg = result["messages"][-1]
        response_text = msg_content(ai_msg)

        print(f"\nAssistant: {response_text}\n")

        strategy.add_agent_response(result["messages"])


if __name__ == "__main__":
    main()
