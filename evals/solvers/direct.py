"""
evals/solvers/direct.py

Direct solver: no agent loop, no files. Just sends the raw query to the model
and returns the response. Serves as the unstructured baseline.
"""

from inspect_ai.solver import Solver, chain, generate, system_message


def direct_solver() -> Solver:
    """Plain generate() baseline — no agent, no files."""
    return chain(
        system_message("You are a helpful assistant."),
        generate(),
    )
