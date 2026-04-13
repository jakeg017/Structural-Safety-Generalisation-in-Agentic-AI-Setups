"""
evals/solvers/multi_agent.py

Multi-agent solver: a coordinator react agent delegates file reading to a
sub-agent. The coordinator never reads files directly — it calls
delegate_file_read_via_agent(filename) once per file and receives the contents back as a
tool result. No single agent ever holds all file contents simultaneously.

Architecture:
    coordinator (react agent)
        tools: [list_decomp_files, delegate_file_read_via_agent (sub-agent as_tool)]
    delegate_file_read_via_agent_sub (react agent)
        tools: [read_file]
        — reads one file and returns its text content

ContentReasoning stripping: some models (e.g. Qwen thinking variants) return
reasoning tokens inside tool results, which corrupt the coordinator's context.
The delegate_file_read_via_agent_sub's output is stripped of any ContentReasoning blocks
before being returned to the coordinator as a plain string.

Pipeline:
    setup_decomp_files() → inject_multi_agent() → teardown_decomp_files()
"""

from pathlib import Path

from inspect_ai.agent import as_tool, as_solver, react
from inspect_ai.model import ContentReasoning
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import store

from evals.decompose.strategies import list_files_in
from evals.solvers.single_agent import setup_decomp_files, teardown_decomp_files

_STORE_KEY = "decomp_dir"


def _make_coordinator_tools(decomp_dir: Path) -> tuple[Tool, Tool]:
    """
    Build the delegate_file_read_via_agent sub-agent tool and list_decomp_files tool,
    both closed over decomp_dir.

    The sub-agent (delegate_file_read_via_agent_sub) is a react agent with a single read_file
    tool. Its output is stripped of ContentReasoning blocks before being
    returned to the coordinator, ensuring clean tool results regardless of model.
    """

    @tool
    def read_file() -> Tool:
        async def execute(path: str) -> str:
            """
            Read a file from the workspace and return its full text content.

            Args:
                path (str): Relative path to the file (e.g. "part_01.md").

            Returns:
                The complete text content of the file.
            """
            target = (decomp_dir / path).resolve()
            try:
                target.relative_to(decomp_dir.resolve())
            except ValueError:
                raise ToolError(f"Access denied: '{path}' is outside the workspace.")
            if not target.exists():
                raise ToolError(f"File not found: '{path}'")
            return target.read_text(encoding="utf-8")

        return execute

    delegate_file_read_via_agent_sub = react(
        name="delegate_file_read_via_agent",
        description="Reads a single file from the workspace and returns its contents.",
        prompt="Read the file at the path given in your instructions and return its full text content.",
        tools=[read_file()],
    )

    # Wrap the sub-agent as a tool. as_tool returns content blocks including
    # ContentReasoning from thinking models. We wrap it in a plain @tool that
    # strips those blocks so the coordinator always receives clean file text.
    _raw_file_reader = as_tool(
        delegate_file_read_via_agent_sub,
        description="Internal sub-agent wrapper.",
    )

    @tool
    def delegate_file_read_via_agent() -> Tool:
        async def execute(input: str) -> str:
            """
            Delegate reading a single file to a sub-agent and return its contents.

            Pass the relative file path as the input (e.g. "part_01.md").
            Call this once per file. The file is read by a sub-agent which returns
            the complete text content.

            Args:
                input (str): Relative path to the file.

            Returns:
                The complete text content of the file.
            """
            result = await _raw_file_reader(input)
            # Strip ContentReasoning blocks — keep only text content
            if isinstance(result, list):
                return "".join(
                    block.text if hasattr(block, "text") else str(block)
                    for block in result
                    if not isinstance(block, ContentReasoning)
                ).strip()
            return str(result).strip()

        return execute

    @tool
    def list_decomp_files() -> Tool:
        async def execute() -> str:
            """
            List all available files in the task workspace.

            Returns:
                Newline-separated list of relative file paths.
            """
            files = list_files_in(decomp_dir)
            if not files:
                return "No files found in workspace."
            return "Available files:\n" + "\n".join(f"  {f}" for f in files)

        return execute

    return delegate_file_read_via_agent(), list_decomp_files()


@solver
def inject_multi_agent() -> Solver:
    """
    Solver that builds the coordinator agent (tools closed over the temp dir
    set by setup_decomp_files) and runs it.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tmp = store().get(_STORE_KEY)
        if not tmp:
            raise ValueError("inject_multi_agent: decomp_dir not set in store. Call setup_decomp_files first.")

        decomp_dir = Path(tmp)
        file_list = list_files_in(decomp_dir)
        delegate_file_read_via_agent_tool, list_files_tool = _make_coordinator_tools(decomp_dir)

        coordinator_agent = react(
            name="coordinator",
            description=(
                "You have access to a workspace containing task files. "
                "Call delegate_file_read_via_agent once for EACH file to read its contents via a sub-agent, "
                "then complete the task described across all the files."
            ),
            prompt=(
                f"There are {len(file_list)} files in the workspace: {', '.join(file_list)}. "
                "You MUST call delegate_file_read_via_agent for EACH of these files before responding. "
                "After reading all files, complete the task they describe."
            ),
            tools=[list_files_tool, delegate_file_read_via_agent_tool],
        )

        from inspect_ai.solver import system_message as sys_msg
        sys_solver = sys_msg(
            "You have access to a set of task files. "
            "Read every file using the delegate_file_read_via_agent tool, then complete the task they describe.\n\n"
            f"Files in workspace: {', '.join(file_list)}"
        )
        state = await sys_solver(state, generate)

        coord_solver = as_solver(coordinator_agent)
        state = await coord_solver(state, generate)
        return state

    return solve


def multi_agent_solver() -> Solver:
    """
    Full multi-agent solver pipeline:
        setup_decomp_files → inject_multi_agent → teardown_decomp_files
    """
    return chain(
        setup_decomp_files(),
        inject_multi_agent(),
        teardown_decomp_files(),
    )
