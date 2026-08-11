"""Agent-boundary tests: the Bedrock model is not caller-selectable.

The agent used to accept a per-invocation ``model_id`` off the request body and
let it override the configured model, which is why the Runtime role's Bedrock
grant had to cover every foundation model in the account. The override is gone;
these tests lock that in so it cannot quietly return.

Why these assertions are AST-based rather than an HTTP test
-----------------------------------------------------------
``agent/main.py`` imports ``fastapi`` and ``agent_core`` imports ``strands`` /
``mcp`` — the agent's *container* dependencies, which are deliberately NOT in
``requirements-dev.txt`` (the dev venv installs only pytest/hypothesis/moto/boto3,
see the file's header). Importing either module here would make the whole test
suite depend on the agent image's runtime stack.

Parsing the source with ``ast`` instead keeps the check dependency-free while
still being a real structural assertion, not a substring grep: it walks the
actual call/parameter/assignment nodes, so a rename or a differently-spelled
re-introduction of the override is still caught.
"""

from __future__ import annotations

import ast
import functools
import os

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "agent")

_MAIN_PATH = os.path.join(_AGENT_DIR, "main.py")
_CORE_PATH = os.path.join(_AGENT_DIR, "agent_core.py")

# The request-body variable names ``main.py`` parses the invocation payload from.
_BODY_NAMES = frozenset({"body", "payload"})


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
    """Locate a top-level (sync or async) function definition by name.

    Args:
        tree: The parsed module.
        name: The function name to find.

    Returns:
        The matching function definition node.

    Raises:
        AssertionError: If no such function exists in the module.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _subscript_keys_read_from_body(tree: ast.Module) -> set[str]:
    """Collect every constant key read out of the request body in a module.

    Catches both ``body.get("k")`` / ``body["k"]`` and the nested
    ``body["input"].get("k")`` shape the legacy payload used.

    Args:
        tree: The parsed module.

    Returns:
        The set of string keys read from a request-body variable.
    """
    keys: set[str] = set()

    for node in ast.walk(tree):
        # body["k"] / body["input"]["k"]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.value, ast.Name) and node.value.id in _BODY_NAMES:
                keys.add(str(node.slice.value))
            if isinstance(node.value, ast.Subscript):
                inner = node.value.value
                if isinstance(inner, ast.Name) and inner.id in _BODY_NAMES:
                    keys.add(str(node.slice.value))
        # body.get("k") / body["input"].get("k")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "get" or not node.args:
                continue
            first = node.args[0]
            if not isinstance(first, ast.Constant):
                continue
            target = node.func.value
            if isinstance(target, ast.Name) and target.id in _BODY_NAMES:
                keys.add(str(first.value))
            elif isinstance(target, ast.Subscript):
                inner = target.value
                if isinstance(inner, ast.Name) and inner.id in _BODY_NAMES:
                    keys.add(str(first.value))
    return keys


def test_invocation_body_is_not_read_for_a_model_id() -> None:
    """``main.py`` must not read any model selector off the invocation payload."""
    keys = _subscript_keys_read_from_body(_parse(_MAIN_PATH))

    assert "model_id" not in keys
    assert not any("model" in key for key in keys), (
        f"a model selector is being read from the request body: {sorted(keys)}"
    )
    # The keys the endpoint legitimately reads are unchanged.
    assert {"prompt", "user_jwt", "input"} <= keys


def test_request_model_declares_no_model_field() -> None:
    """The pydantic request model exposes no model selector either.

    Reading a model id off a declared ``InvocationRequest`` field instead of the
    raw body dict would bypass the body-key walk above, so the class body is
    checked directly.
    """
    tree = _parse(_MAIN_PATH)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InvocationRequest"
    ]
    assert classes, "InvocationRequest is no longer declared in main.py"

    fields = {
        target.id
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in [node.target]
    }
    assert not any("model" in field for field in fields), (
        f"a model selector is declared on InvocationRequest: {sorted(fields)}"
    )


def test_create_agent_and_invoke_takes_no_model_parameter() -> None:
    """The invocation entrypoint exposes no model override parameter."""
    func = _function(_parse(_CORE_PATH), "create_agent_and_invoke")
    args = func.args
    names = {
        a.arg
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
    }

    assert "model_id" not in names
    assert not any("model" in name for name in names), (
        f"create_agent_and_invoke still accepts a model selector: {sorted(names)}"
    )
    assert {"session_id", "prompt", "user_jwt"} <= names


def _invocation_call_sites(tree: ast.Module) -> list[ast.Call]:
    """Return the call nodes that carry arguments through to the agent entrypoint.

    Two shapes count, because ``main.py`` no longer calls the entrypoint directly:

    * a direct ``create_agent_and_invoke(...)`` call; and
    * an offload — ``asyncio.to_thread(create_agent_and_invoke, ...)`` — which is
      the current shape, since a synchronous run on the event loop would block
      ``GET /ping`` for the whole invocation. Here the keyword arguments
      destined for the entrypoint sit on the *offload* call, so that is the node
      whose keywords must be inspected.

    Accepting both keeps this module's model-pinning guarantee independent of how
    the run is scheduled.

    Args:
        tree: Parsed ``main.py`` AST.

    Returns:
        Every call node whose keywords are forwarded to the entrypoint.
    """
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Shape 1: called directly.
        if isinstance(node.func, ast.Name) and node.func.id == "create_agent_and_invoke":
            sites.append(node)
            continue
        # Shape 2: passed by name to a thread offload, which forwards the kwargs.
        callee = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        if callee in {"to_thread", "run_in_threadpool"} and any(
            isinstance(arg, ast.Name) and arg.id == "create_agent_and_invoke"
            for arg in node.args
        ):
            sites.append(node)
    return sites


def test_main_passes_no_model_argument_to_the_entrypoint() -> None:
    """The ``create_agent_and_invoke`` call site forwards no model selector."""
    calls = _invocation_call_sites(_parse(_MAIN_PATH))
    assert calls, "create_agent_and_invoke is no longer invoked from main.py"

    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "model_id" not in keywords
        assert not any(kw and "model" in kw for kw in keywords), (
            f"model selector forwarded from main.py: {sorted(k for k in keywords if k)}"
        )


def test_bedrock_model_is_sourced_only_from_config() -> None:
    """``effective_model_id`` is assigned from the config dict, unconditionally.

    An ``if``/ternary in that assignment is exactly the shape the removed
    override had, so the value must come from a plain ``config[...]`` subscript.
    """
    func = _function(_parse(_CORE_PATH), "create_agent_and_invoke")

    assignments = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "effective_model_id"
            for t in node.targets
        )
    ]
    assert len(assignments) == 1, "expected exactly one effective_model_id assignment"

    value = assignments[0].value
    assert isinstance(value, ast.Subscript), (
        "effective_model_id must be read straight from config, not conditionally "
        f"resolved (got {type(value).__name__})"
    )
    assert isinstance(value.value, ast.Name) and value.value.id == "config"
    assert isinstance(value.slice, ast.Constant)
    assert value.slice.value == "bedrock_model_id"

    # And it is what actually reaches the model provider.
    bedrock_calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "BedrockModel"
    ]
    assert len(bedrock_calls) == 1
    model_kwarg = {
        kw.arg: kw.value for kw in bedrock_calls[0].keywords
    }.get("model_id")
    assert isinstance(model_kwarg, ast.Name)
    assert model_kwarg.id == "effective_model_id"
