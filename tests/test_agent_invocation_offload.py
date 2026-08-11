"""Agent-boundary tests: the agent run never blocks the event loop.

``POST /invocations`` used to call ``create_agent_and_invoke`` inline from an
``async def`` handler. That function is fully synchronous and drives the Strands
agent to completion — up to ``MAX_AGENT_TURNS`` model turns — so the coroutine
held the event loop for the entire run. Nothing else on that worker could be
served meanwhile, including ``GET /ping``: the container could not report
liveness exactly while it was busy doing the work it exists for, which is why
``agent/Dockerfile`` ships no ``HEALTHCHECK``. These tests lock the
offload in so the inline call cannot quietly return.

Why these assertions are AST-based rather than an HTTP test
-----------------------------------------------------------
The obvious test — hold an invocation open and assert ``GET /ping`` still
answers 200 — needs an ASGI client driving the real app, so ``fastapi`` plus
``httpx`` plus an async test plugin, and ``agent_core`` would drag in
``strands`` / ``mcp``. None of those are in ``requirements-dev.txt``: the dev
venv installs only pytest/hypothesis/moto/boto3/PyJWT, and ``conftest.py`` says
so explicitly. Adding the agent image's whole runtime stack to the test
toolchain to assert one scheduling property is a bad trade.

Parsing the source with ``ast`` keeps the check dependency-free while remaining
a real structural assertion rather than a substring grep: it walks the actual
call nodes, so a rename, a re-indent, or a differently-spelled reintroduction of
the inline call is still caught. This mirrors
``tests/test_agent_model_pinning.py``, which takes the same approach for the
same reason.

What this cannot catch: whether the offload *works at runtime*. That was
verified by building the image and exercising the endpoint; see the PR.
"""

from __future__ import annotations

import ast
import functools
import os

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "agent")
_MAIN_PATH = os.path.join(_AGENT_DIR, "main.py")

#: The synchronous entrypoint that must never be awaited inline on the loop.
_BLOCKING_CALLEE = "create_agent_and_invoke"

#: Accepted offload idioms. ``asyncio.to_thread`` is what the code uses;
#: starlette's ``run_in_threadpool`` is an equally correct alternative, so the
#: test does not force a rewrite if someone switches between them.
_OFFLOAD_NAMES = frozenset({"to_thread", "run_in_threadpool"})


@functools.lru_cache(maxsize=None)
def _parse(path: str) -> ast.Module:
    """Parse a source file into an AST (cached across assertions).

    Args:
        path: Absolute path to the Python source file.

    Returns:
        The parsed module AST.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the top-level function or coroutine named ``name``.

    Args:
        tree: Parsed module AST.
        name: Function name to find.

    Returns:
        The matching function definition node.

    Raises:
        AssertionError: If no function of that name exists.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_MAIN_PATH}")


def _callee_name(node: ast.expr) -> str:
    """Return the trailing identifier of a call target.

    ``asyncio.to_thread`` -> ``to_thread``; a bare ``foo`` -> ``foo``.

    Args:
        node: The ``func`` expression of an ``ast.Call``.

    Returns:
        The final identifier, or the empty string for a shape with none.
    """
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _offload_calls(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    """Return the offload calls (``to_thread`` / ``run_in_threadpool``) inside ``fn``."""
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _callee_name(node.func) in _OFFLOAD_NAMES
    ]


def test_invocation_handler_is_a_coroutine() -> None:
    """``invoke_agent`` must stay ``async def``.

    It awaits ``parse_bounded_json`` for the bounded body read, so it cannot be
    demoted to a plain ``def`` (which is the other way to get FastAPI to run a
    handler off the loop). Since it must remain a coroutine, the offload of the
    blocking agent run has to be explicit — that is what the tests below pin.
    """
    fn = _function(_parse(_MAIN_PATH), "invoke_agent")
    assert isinstance(fn, ast.AsyncFunctionDef), (
        "invoke_agent is no longer a coroutine; if that is deliberate, the "
        "offload requirement in this module needs revisiting"
    )


def test_agent_run_is_never_invoked_inline() -> None:
    """The blocking run must not appear as a direct call anywhere in the handler.

    This is the regression guard. ``create_agent_and_invoke`` may be *referenced*
    (passed to the offload as a callable) but never *called* inside the handler
    body, because a direct call executes on the event loop.
    """
    fn = _function(_parse(_MAIN_PATH), "invoke_agent")
    direct = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _callee_name(node.func) == _BLOCKING_CALLEE
    ]
    assert not direct, (
        f"{_BLOCKING_CALLEE} is called directly in invoke_agent (line "
        f"{direct[0].lineno}); it blocks the event loop for the whole agent run. "
        f"Pass it to asyncio.to_thread instead."
    )


def test_agent_run_is_offloaded_and_awaited() -> None:
    """The run is handed to a worker thread, and the result is awaited.

    Guards two distinct mistakes: forgetting the offload entirely, and the subtle
    ``asyncio.to_thread(create_agent_and_invoke(...))`` — which evaluates the
    blocking call eagerly on the loop and offloads its *result*, defeating the
    fix while still mentioning ``to_thread``.
    """
    fn = _function(_parse(_MAIN_PATH), "invoke_agent")
    offloads = _offload_calls(fn)
    assert offloads, (
        "invoke_agent performs no thread offload; the synchronous agent run must "
        "go through asyncio.to_thread (or run_in_threadpool)."
    )

    carrying = [
        call
        for call in offloads
        # The callable must be passed by NAME, not called: to_thread(fn, ...) not
        # to_thread(fn(...)). An ast.Name arg is the former; an ast.Call is the bug.
        if any(
            isinstance(arg, ast.Name) and arg.id == _BLOCKING_CALLEE
            for arg in call.args
        )
    ]
    assert carrying, (
        f"no offload call passes {_BLOCKING_CALLEE} as a bare callable. Check for "
        f"to_thread({_BLOCKING_CALLEE}(...)), which runs it on the loop first and "
        f"offloads only the return value."
    )

    awaited = {
        id(node.value)
        for node in ast.walk(fn)
        if isinstance(node, ast.Await)
    }
    assert any(id(call) in awaited for call in carrying), (
        f"the {_BLOCKING_CALLEE} offload is not awaited; an un-awaited coroutine "
        f"never runs and the handler would return before the agent does."
    )


def test_offload_keeps_the_invocation_arguments() -> None:
    """The offload forwards the same arguments the inline call used.

    A mechanical guard against a partially-applied refactor: dropping
    ``user_jwt`` here would silently strip the caller's identity from the agent
    run, which the Gateway needs for the MCP session.
    """
    fn = _function(_parse(_MAIN_PATH), "invoke_agent")
    forwarded = {
        kw.arg
        for call in _offload_calls(fn)
        for kw in call.keywords
        if kw.arg is not None
    }
    for expected in ("session_id", "prompt", "user_jwt"):
        assert expected in forwarded, (
            f"the offload no longer forwards {expected!r} to {_BLOCKING_CALLEE}"
        )
