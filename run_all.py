"""
run_all.py

Run all 11 eval conditions in a single call.

Conditions:
    1.  direct          × none
    2.  single_agent    × simple_split
    3.  single_agent    × role_based
    4.  single_agent    × nested
    5.  single_agent    × breadcrumb
    6.  single_agent    × mixed_benign
    7.  multi_agent     × simple_split
    8.  multi_agent     × role_based
    9.  multi_agent     × nested
    10. multi_agent     × breadcrumb
    11. multi_agent     × mixed_benign

Usage:
    python run_all.py [--model MODEL] [--base-url URL] [--num-goals N] [--judge JUDGE_MODEL]

Defaults:
    --model      openai/deepseek-chat
    --base-url   https://api.deepseek.com/v1
    --num-goals  10
    --judge      openai/gpt-5.4-mini

Logs are saved to logs/ and can be viewed with:
    inspect view --log-dir logs/
or:
    python view_logs.py
"""

import argparse
import os

from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model

from evals.tasks.eval_task import eval_task

_STRATEGIES = ["simple_split", "role_based", "nested", "breadcrumb", "mixed_benign"]

_CONDITIONS = [("direct", "none")] + [
    (runner, strategy)
    for runner in ["single_agent", "multi_agent"]
    for strategy in _STRATEGIES
]


def main():
    parser = argparse.ArgumentParser(description="Run all eval conditions.")
    parser.add_argument("--model", default="openai/deepseek-chat", help="Model under test")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1", help="Custom base URL for model")
    parser.add_argument("--num-goals", type=int, default=10, help="Goals per condition")
    parser.add_argument("--num-files", type=int, default=4, help="Files per decomposition")
    parser.add_argument("--epochs", type=int, default=1, help="Runs per sample (3 matches SSG paper methodology)")
    parser.add_argument("--max-connections", type=int, default=None, help="Max concurrent API connections (use to respect provider rate limits)")
    parser.add_argument("--judge", default="openai/gpt-5.4-mini", help="StrongREJECT judge model")
    parser.add_argument("--log-dir", default="logs/", help="Directory for eval logs")
    args = parser.parse_args()

    tasks = [
        eval_task(
            runner=runner,
            strategy=strategy,
            num_goals=args.num_goals,
            num_files=args.num_files,
            judge_model=args.judge,
            epochs=args.epochs,
        )
        for runner, strategy in _CONDITIONS
    ]

    print(f"Running {len(tasks)} conditions x {args.num_goals} goals x {args.epochs} epoch(s)")
    print(f"Model: {args.model}  |  Base URL: {args.base_url or '(default)'}  |  Judge: {args.judge}  |  Logs: {args.log_dir}\n")

    # inspect_ai's OpenAI provider checks for OPENAI_API_KEY on initialisation
    # even when api_key is passed explicitly. Use JUDGE_API_KEY to satisfy this
    # check — the actual DeepSeek calls use MODEL_API_KEY via the api_key arg.
    if os.environ.get("JUDGE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["JUDGE_API_KEY"]

    config = GenerateConfig(max_connections=args.max_connections) if args.max_connections else GenerateConfig()
    model = get_model(args.model, base_url=args.base_url, api_key=os.environ.get("MODEL_API_KEY"), config=config) if args.base_url else args.model
    eval(tasks, model=model, log_dir=args.log_dir, fail_on_error=False, timeout=240)


if __name__ == "__main__":
    main()
