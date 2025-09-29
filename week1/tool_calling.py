import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable

from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 3


# ==========================
# Tool implementation (the "executor")
# ==========================
def _annotation_to_str(annotation: Optional[ast.AST]) -> str:
    if annotation is None:
        return "None"
    try:
        return ast.unparse(annotation)  # type: ignore[attr-defined]
    except Exception:
        # Fallback best-effort
        if isinstance(annotation, ast.Name):
            return annotation.id
        return type(annotation).__name__


def _list_function_return_types(file_path: str) -> List[Tuple[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    results: List[Tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return_str = _annotation_to_str(node.returns)
            results.append((node.name, return_str))
    # Sort for stable output
    results.sort(key=lambda x: x[0])
    return results


def output_every_func_return_type(file_path: str = None) -> str:
    """Tool: Return a newline-delimited list of "name: return_type" for each top-level function."""
    path = file_path or __file__
    if not os.path.isabs(path):
        # Try file relative to this script if not absolute
        candidate = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(candidate):
            path = candidate
    pairs = _list_function_return_types(path)
    return "\n".join(f"{name}: {ret}" for name, ret in pairs)


# Sample functions to ensure there is something to analyze
def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    return f"Hello, {name}!"

# Tool registry for dynamic execution by name
TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "output_every_func_return_type": output_every_func_return_type,
    # "add": add,
    # "greet": greet,
}

# ==========================
# Prompt scaffolding
# ==========================

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = f"""
    You are a tool calling assistant whose goal is to execute a provided tool.
    
    Here are the tools you can execute:
    {list(TOOL_REGISTRY.keys())}

Output exactly one JSON object (not a string). 
Do NOT wrap the JSON in quotes or backticks; no code fences; no prose.

WRONG (do not do):
"{{\\"tool\\": \\"output_every_func_return_type\\", \\"args\\": {{}}}}"

RIGHT (output exactly this):
{{"tool": "output_every_func_return_type", "args": {{}}}}
"""


def resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    here = os.path.dirname(__file__)
    c1 = os.path.join(here, p)
    if os.path.exists(c1):
        return c1
    # Try sibling of project root if needed
    return p


def extract_tool_call(text: str) -> Dict[str, Any]:
    """Parse a single JSON object from the model output."""
    text = text.strip()
    # Some models wrap JSON in code fences; attempt to strip
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json\n"):
            text = text[5:]
    try:
        obj = json.loads(text)
        return obj
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON for the tool call")


def run_model_for_tool_call(system_prompt: str) -> Dict[str, Any]:
    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Call the tool now."},
        ],
        options={"temperature": 0.3},
    )
    content = response.message.content
    return extract_tool_call(content)


def execute_tool_call(call: Dict[str, Any]) -> str:
    name = call.get("tool")
    if not isinstance(name, str):
        raise ValueError("Tool call JSON missing 'tool' string")
    func = TOOL_REGISTRY.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Tool call JSON 'args' must be an object")

    # Best-effort path resolution if a file_path arg is present
    if "file_path" in args and isinstance(args["file_path"], str):
        args["file_path"] = resolve_path(args["file_path"]) if str(args["file_path"]) != "" else __file__
    elif "file_path" not in args:
        # Provide default for tools expecting file_path
        args["file_path"] = __file__

    return func(**args)


def compute_expected_output() -> str:
    # Ground-truth expected output based on the actual file contents
    return output_every_func_return_type(__file__)


def test_your_prompt(system_prompt: str) -> bool:
    """Run once: require the model to produce a valid tool call; compare tool output to expected."""
    expected = compute_expected_output()
    for _ in range(NUM_RUNS_TIMES):
        try:
            call = run_model_for_tool_call(system_prompt)
        except Exception as exc:
            print(f"Failed to parse tool call: {exc}")
            continue
        try:
            actual = execute_tool_call(call)
        except Exception as exc:
            print(f"Tool execution failed: {exc}")
            continue
        if actual.strip() == expected.strip():
            print(f">>> Generated tool call: \n{call}\n")
            print(f">>> Generated output: \n{actual}\n")
            print("SUCCESS")
            return True
        else:
            print("Expected output:\n" + expected)
            print("  Actual output:\n" + actual)
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)
