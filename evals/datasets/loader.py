"""
evals/datasets/loader.py

Load harmful goals from the dataset CSV.

Columns used: "Goal ID" (UUID) and "Goal" (plain-language harmful intent).

We use the "Goal" field — not the jailbreak "Prompt" column — because the
harness constructs its own structural variants from the raw goal text.
"""

import csv
from pathlib import Path
from typing import Optional

_DEFAULT_CSV = (
    Path(__file__).parent.parent.parent
    / "datasets"
    / "Harmful Dataset.csv"
)


def load_goals(
    csv_path: Path | str = _DEFAULT_CSV,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Load harmful goals from the dataset CSV.

    Returns:
        list of dicts with keys:
            "id"    — Goal ID (UUID string)
            "query" — Goal text (plain-language harmful intent)

    Args:
        csv_path: Path to the CSV. Defaults to datasets/Harmful Dataset.csv.
        limit:    If set, return only the first N unique goals.
    """
    goals: list[dict] = []
    seen_ids: set[str] = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goal_id = row.get("Goal ID", "").strip()
            goal_text = row.get("Goal", "").strip()
            if goal_id and goal_text and goal_id not in seen_ids:
                seen_ids.add(goal_id)
                goals.append({"id": goal_id, "query": goal_text})
            if limit and len(goals) >= limit:
                break

    return goals
