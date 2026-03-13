"""
StreamStatus — visual feedback for model streaming.

Phase 1 (waiting):   Braille spinner with elapsed time on the console line.
Phase 2 (streaming): Live token count + t/s in the terminal title bar.
Phase 3 (done):      Dim summary line, terminal title reset.
"""

import sys
import threading
import time

DIM = "\033[90m"
RESET = "\033[0m"

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StreamStatus:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._start_time = None
        self._first_token_time = None
        self._token_count = 0
        self._prefix = ""

    # -- Phase 1: spinner --------------------------------------------------

    def start(self, prefix=""):
        """Show an animated spinner with elapsed time.

        Any leading newlines in prefix are printed once upfront so the
        spinner thread can use \\r to overwrite the same line.
        """
        # Print leading newlines once, keep the rest for \r overwrites
        stripped = prefix.lstrip("\n")
        leading = prefix[: len(prefix) - len(stripped)]
        if leading:
            sys.stdout.write(leading)
            sys.stdout.flush()
        self._prefix = stripped
        self._start_time = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._start_time
            frame = _BRAILLE[i % len(_BRAILLE)]
            sys.stdout.write(f"\r\033[K{self._prefix}{frame} {elapsed:.1f}s")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    # -- Phase 2: streaming tokens -----------------------------------------

    def first_token(self):
        """Stop the spinner and print the real header."""
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        self._first_token_time = time.monotonic()
        # Clear spinner line then print header
        sys.stdout.write(f"\r\033[K{self._prefix}")
        sys.stdout.flush()

    def count_token(self):
        """Increment token counter and update terminal title (throttled)."""
        self._token_count += 1
        if self._token_count % 5 == 0:
            elapsed = time.monotonic() - (self._first_token_time or self._start_time)
            tps = self._token_count / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\033]0;{self._token_count} tokens \u00b7 {tps:.1f} t/s\007")
            sys.stdout.flush()

    # -- Phase 3: done -----------------------------------------------------

    def finish(self):
        """Reset terminal title and print summary stats."""
        # Stop spinner if still running (e.g. cancelled before first token)
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None

        # Reset terminal title
        sys.stdout.write("\033]0;\007")
        sys.stdout.flush()

        if self._start_time is None:
            return

        total = time.monotonic() - self._start_time
        if self._token_count > 0:
            tps = self._token_count / total if total > 0 else 0
            print(f"{DIM}[{self._token_count} tokens \u00b7 {total:.1f}s \u00b7 {tps:.1f} t/s]{RESET}")
        else:
            # Spinner was shown but no tokens arrived (e.g. tool-call-only)
            sys.stdout.write(f"\r\033[K")
            sys.stdout.flush()
