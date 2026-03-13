"""Search files tool — grep through files for patterns."""

import os
import re
from pathlib import Path


_MAX_RESULTS = 100


def fn(pattern: str, path: str = ".", glob: str = "*", ignore_case: bool = True) -> str:
    """Search file contents for a regex pattern.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in (default: current directory).
        glob: File glob pattern to filter (default: * for all files).
        ignore_case: Case-insensitive search (default: True).
    """
    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    search_path = Path(path)
    if not search_path.exists():
        return f"Error: path '{path}' does not exist"

    results = []
    files_searched = 0
    files_matched = 0

    for file_path in sorted(search_path.rglob(glob)):
        if not file_path.is_file():
            continue
        # Skip binary files, hidden dirs, and common noise
        rel = str(file_path.relative_to(search_path))
        if any(part.startswith(".") for part in file_path.parts if part != "."):
            continue
        if "__pycache__" in rel or "node_modules" in rel:
            continue

        files_searched += 1
        try:
            text = file_path.read_text(errors="ignore")
        except Exception:
            continue

        matched_in_file = False
        for line_num, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                if not matched_in_file:
                    files_matched += 1
                    matched_in_file = True
                results.append(f"{rel}:{line_num}: {line.rstrip()}")
                if len(results) >= _MAX_RESULTS:
                    break
        if len(results) >= _MAX_RESULTS:
            break

    header = f"[Searched {files_searched} files, {files_matched} matched, {len(results)} results"
    if len(results) >= _MAX_RESULTS:
        header += " (truncated)"
    header += "]\n"

    if not results:
        return header + "No matches found."

    return header + "\n".join(results)


definition = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": (
            "Search file contents for a regex pattern (like grep). "
            "Searches recursively through a directory. Use this to find patterns "
            "in code, search memory files, review past cycle logs, or locate "
            "specific content across the project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory).",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob to filter, e.g. '*.py', '*.json', '*.md' (default: all files).",
                    "default": "*",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: true).",
                    "default": True,
                },
            },
            "required": ["pattern"],
        },
    },
}
