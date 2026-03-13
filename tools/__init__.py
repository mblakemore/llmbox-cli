"""
Tool registry with auto-discovery.

Each tool module in this directory should export:
  - fn: the callable implementation
  - definition: the OpenAI-compatible tool schema dict

To add a new tool, create a new .py file in this directory following that convention.
"""

import importlib
import importlib.util
import logging
import pkgutil
from pathlib import Path

MAP_FN = {}
tools = []

_log = logging.getLogger("agent")


def _discover_tools():
    """Auto-discover and register all tool modules in this package."""
    package_dir = Path(__file__).parent
    for finder, name, ispkg in pkgutil.iter_modules([str(package_dir)]):
        module = importlib.import_module(f".{name}", package=__package__)
        if hasattr(module, "fn") and hasattr(module, "definition"):
            tool_name = module.definition["function"]["name"]
            MAP_FN[tool_name] = module.fn
            tools.append(module.definition)


def load_extra_tools(directory):
    """Load agent-specific tools from an external directory.

    Discovers .py files in the given directory (skipping _-prefixed files),
    loads each via importlib, and registers fn + definition if present.
    Agent tools override shared tools of the same name.

    Args:
        directory: Path to directory containing tool .py files.
    """
    tool_dir = Path(directory)
    if not tool_dir.is_dir():
        return

    for py_file in sorted(tool_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"extra_tools.{py_file.stem}", str(py_file))
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "fn") and hasattr(module, "definition"):
                tool_name = module.definition["function"]["name"]
                # Override shared tool if same name exists
                if tool_name in MAP_FN:
                    # Remove old definition from tools list
                    for i, t in enumerate(tools):
                        if t["function"]["name"] == tool_name:
                            tools[i] = module.definition
                            break
                else:
                    tools.append(module.definition)
                MAP_FN[tool_name] = module.fn
                _log.debug("Loaded extra tool: %s from %s", tool_name, py_file.name)
        except Exception as e:
            _log.warning("Failed to load extra tool %s: %s", py_file.name, e)


_discover_tools()
