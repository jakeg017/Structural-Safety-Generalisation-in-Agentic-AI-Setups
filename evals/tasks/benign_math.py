"""
evals/tasks/benign_math.py

Benign math smoke-test task.

Purpose: validate the full inspect_ai pipeline (dataset → solver → scorer →
logging → viewer) using harmless arithmetic problems before running any
harmful content evals.

Run:
    inspect eval evals/tasks/benign_math.py --model openai/gpt-4o-mini --log-dir logs/

Then view:
    inspect view --log-dir logs/
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import chain, generate, system_message

_SAMPLES = [
    Sample(
        id="math_1",
        input="What is 347 × 28?",
        target="9716",
    ),
    Sample(
        id="math_2",
        input="What is 1024 ÷ 32?",
        target="32",
    ),
    Sample(
        id="math_3",
        input="What is 17² − 144?",
        target="145",
    ),
    Sample(
        id="math_4",
        input="A train travels 270 km in 3 hours. What is its average speed in km/h?",
        target="90",
    ),
    Sample(
        id="math_5",
        input="What is 15% of 840?",
        target="126",
    ),
]


@task
def benign_math() -> Task:
    """Five arithmetic problems — pipeline smoke test."""
    return Task(
        dataset=MemoryDataset(_SAMPLES),
        solver=chain(
            system_message("You are a helpful math assistant. Answer concisely with just the number."),
            generate(),
        ),
        scorer=model_graded_qa(
            instructions=(
                "The question is a math problem. "
                "The answer is CORRECT if the numeric value in the model response matches "
                "the target (ignoring formatting, units, or extra explanation). "
                "Grade CORRECT or INCORRECT."
            ),
        ),
    )
