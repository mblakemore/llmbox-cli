"""Sleep for a specified duration."""

import time


def fn(seconds: float) -> str:
    """Sleep for the given number of seconds.

    Args:
        seconds: Number of seconds to sleep.
    """
    try:
        time.sleep(seconds)
        return f"Slept for {seconds} seconds"
    except Exception as e:
        return f"Error: {str(e)}"


definition = {
    "type": "function",
    "function": {
        "name": "sleep",
        "description": "Sleep for a specified number of seconds. Useful for waiting between commands or polling for results.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds to sleep.",
                },
            },
            "required": ["seconds"],
        },
    },
}
