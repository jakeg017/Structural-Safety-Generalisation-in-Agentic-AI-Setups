"""
scripts/mann_whitney.py

Mann-Whitney U tests comparing single_agent vs multi_agent scores
for each decomposition strategy, using the most recent log files.

Two metrics are reported:
  - Mean score: continuous StrongREJECT score (0.0–1.0)
  - ASR: Attack Success Rate — fraction of samples where score > 0,
    matching the ceiling-function metric used in the SSG paper.

Usage:
    python3.13 scripts/mann_whitney.py
    python3.13 scripts/mann_whitney.py --log-dir logs/
"""

import argparse
import json
import zipfile
from pathlib import Path

from scipy.stats import mannwhitneyu


def load_scores(log_path: Path) -> tuple[str, str, list[float]]:
    """Extract runner, strategy, and per-sample scores from a log file."""
    with zipfile.ZipFile(log_path) as z:
        header = json.loads(z.read("header.json"))
        task_args = header["eval"]["task_args"]
        runner = task_args["runner"]
        strategy = task_args["strategy"]

        scores = []
        for name in z.namelist():
            if not name.startswith("samples/"):
                continue
            sample = json.loads(z.read(name))
            score = sample.get("scores", {}).get("strongreject", {}).get("value")
            if score is not None:
                scores.append(float(score))

    return runner, strategy, scores


def load_all_logs(log_dir: Path) -> dict[tuple[str, str], list[float]]:
    """
    Load all .eval log files and return a dict keyed by (runner, strategy).
    If multiple logs exist for the same condition, keeps the most recent.
    """
    results: dict[tuple[str, str], tuple[Path, list[float]]] = {}

    for log_path in sorted(log_dir.glob("*.eval")):
        try:
            runner, strategy, scores = load_scores(log_path)
        except Exception as e:
            print(f"  Skipping {log_path.name}: {e}")
            continue

        key = (runner, strategy)
        results[key] = (log_path, scores)

    return {k: v[1] for k, v in results.items()}


def asr(scores: list[float]) -> float:
    """Attack Success Rate: fraction of samples where score > 0."""
    return sum(1 for s in scores if s > 0) / len(scores)


def print_table(title: str, hypothesis: str, get_values, all_scores, strategies):
    print()
    print(title)
    print(f"H1 (directional): {hypothesis}")
    print("=" * 65)
    print(f"{'Strategy':<18} {'SA':>8} {'MA':>8} {'Δ':>7} {'U':>8} {'p':>10} {'sig':>5}")
    print("-" * 65)

    for strategy in strategies:
        sa_scores = all_scores.get(("single_agent", strategy))
        ma_scores = all_scores.get(("multi_agent", strategy))

        if sa_scores is None or ma_scores is None:
            print(f"{strategy:<18}  {'(missing data)':>40}")
            continue

        sa_vals = get_values(sa_scores)
        ma_vals = get_values(ma_scores)

        sa_mean = sum(sa_vals) / len(sa_vals)
        ma_mean = sum(ma_vals) / len(ma_vals)
        delta = ma_mean - sa_mean

        stat, p = mannwhitneyu(sa_vals, ma_vals, alternative="less")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"{strategy:<18} {sa_mean:>8.3f} {ma_mean:>8.3f} {delta:>+7.3f} {stat:>8.1f} {p:>10.4f} {sig:>5}")

    print("-" * 65)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/", help="Directory containing .eval log files")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    all_scores = load_all_logs(log_dir)

    strategies = ["simple_split", "role_based", "nested", "breadcrumb", "mixed_benign"]

    # --- Table 1: Mean continuous score ---
    print_table(
        title="[1] Mean StrongREJECT score: single_agent vs multi_agent",
        hypothesis="single_agent mean score < multi_agent mean score",
        get_values=lambda scores: scores,
        all_scores=all_scores,
        strategies=strategies,
    )

    # --- Table 2: ASR (binary, matching SSG paper) ---
    print_table(
        title="\n[2] ASR (Attack Success Rate, score > 0): single_agent vs multi_agent",
        hypothesis="single_agent ASR < multi_agent ASR",
        get_values=lambda scores: [1.0 if s > 0 else 0.0 for s in scores],
        all_scores=all_scores,
        strategies=strategies,
    )

    # --- Summary ASR table ---
    print()
    print("ASR summary (fraction of samples where score > 0):")
    print(f"{'Condition':<30} {'ASR':>6}")
    print("-" * 38)
    for runner, strategy in [("direct", "none")] + [
        (r, s) for r in ["single_agent", "multi_agent"] for s in strategies
    ]:
        scores = all_scores.get((runner, strategy))
        if scores is None:
            continue
        label = f"{runner}/{strategy}"
        print(f"{label:<30} {asr(scores):>6.3f}")
    print()

    print("Sample sizes:")
    for strategy in strategies:
        sa = all_scores.get(("single_agent", strategy))
        ma = all_scores.get(("multi_agent", strategy))
        print(f"  {strategy:<18} single_agent n={len(sa) if sa else 0}, multi_agent n={len(ma) if ma else 0}")
    print()


if __name__ == "__main__":
    main()
