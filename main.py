"""CLI entry point for the multi-tool medical agent.

    python main.py                                  # interactive chat (OpenAI Agents SDK)
    python main.py --backend langchain              # use LangChain AgentExecutor as the router
    python main.py -q "How many patients smoke?"    # one-shot question
"""
from __future__ import annotations

import argparse

from medagent import config


def make_chat(backend: str):
    if backend == "langchain":
        from medagent.lc_agent import LangChainChat

        return LangChainChat()
    from medagent.agents_sdk import AgentsSDKChat

    return AgentsSDKChat()


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-tool medical AI agent")
    p.add_argument("--backend", choices=["agents", "langchain"], default="agents",
                   help="Main router implementation (default: OpenAI Agents SDK)")
    p.add_argument("-q", "--question", help="Ask a single question and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Show SQL agent reasoning")
    args = p.parse_args()

    if args.verbose:
        config.VERBOSE = True

    chat = make_chat(args.backend)

    def answer(q: str) -> None:
        text, tools = chat.ask(q)
        print(f"\n[tools used: {', '.join(tools) or 'none'}]\n{text}\n")

    if args.question:
        answer(args.question)
        return

    print("Medical multi-tool agent. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
