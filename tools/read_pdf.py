"""Read PDF tool — extract text from PDF files with page range support."""

import fitz  # PyMuPDF


_MAX_PAGES_PER_CALL = 50


def fn(path: str, start_page: int = 1, end_page: int = 0) -> str:
    """Extract text from a PDF file.

    Args:
        path: Path to the PDF file.
        start_page: First page to extract (1-indexed, default: 1).
        end_page: Last page to extract (1-indexed, inclusive). 0 = last page.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        return f"Error opening PDF: {e}"

    total = len(doc)
    if total == 0:
        doc.close()
        return "Error: PDF has no pages"

    # Resolve page range
    start = max(1, start_page)
    end = min(total, end_page) if end_page > 0 else total

    if start > total:
        doc.close()
        return f"Error: start_page ({start}) exceeds page count ({total})"

    # Cap to avoid flooding context
    if end - start + 1 > _MAX_PAGES_PER_CALL:
        end = start + _MAX_PAGES_PER_CALL - 1

    parts = [f"[PDF: {path} | Pages {start}-{end} of {total}]"]
    if end < total:
        parts.append(f"[Use read_pdf with start_page={end + 1} to continue reading]")

    for page_num in range(start - 1, end):
        page = doc[page_num]
        text = page.get_text().strip()
        parts.append(f"\n--- Page {page_num + 1} ---\n{text}")

    doc.close()
    return "\n".join(parts)


definition = {
    "type": "function",
    "function": {
        "name": "read_pdf",
        "description": (
            "Extract text from a PDF file (*.pdf only — do NOT use for .py, .md, .json, or other text files; use the 'file' tool with action='read' for those). "
            "Supports page ranges for large documents. "
            "Returns up to 50 pages per call — use start_page to paginate through "
            "longer PDFs. Use this for reading books, papers, and documents for learning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file.",
                },
                "start_page": {
                    "type": "integer",
                    "description": "First page to extract (1-indexed, default: 1).",
                    "default": 1,
                },
                "end_page": {
                    "type": "integer",
                    "description": "Last page to extract (1-indexed, inclusive). 0 = last page.",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
    },
}
