"""
Double-escape cancellation for streaming responses.

Two Escape presses within 400ms aborts the current streaming operation.
Single escapes and ANSI sequences (arrow keys, etc.) are ignored.
"""

import atexit
import select
import sys
import termios
import threading
import time
import tty


class CancelledError(Exception):
    """Raised when the user cancels a streaming operation."""
    pass


_cancel_event = threading.Event()
_monitor_active = threading.Event()
_original_termios = None
_tui_mode = False


def set_tui_mode():
    """Disable cbreak-based cancellation (prompt_toolkit manages the terminal)."""
    global _tui_mode
    _tui_mode = True


def is_cancelled():
    """Check whether cancellation has been requested."""
    return _cancel_event.is_set()


def check_cancelled():
    """Raise CancelledError if cancellation has been requested."""
    if _cancel_event.is_set():
        raise CancelledError()


def reset():
    """Clear the cancellation flag."""
    _cancel_event.clear()


class cbreak_mode:
    """Context manager that switches the terminal to cbreak mode.

    Individual keypresses become readable, Ctrl+C still raises KeyboardInterrupt.
    Original terminal settings are restored on exit.
    """

    def __init__(self):
        self.saved = None

    def __enter__(self):
        global _original_termios
        if not sys.stdin.isatty():
            return self
        self.saved = termios.tcgetattr(sys.stdin)
        _original_termios = self.saved
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        global _original_termios
        if self.saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)
            _original_termios = None
        return False


def _read_byte(timeout):
    """Read a single byte from stdin with timeout. Returns byte or None."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


def _consume_ansi_sequence():
    """Consume the remainder of an ANSI escape sequence after ESC."""
    ch = _read_byte(0.05)
    if ch is None:
        return  # bare ESC, not an ANSI sequence
    if ch == '[':
        # CSI sequence: read until a letter (@ through ~)
        while True:
            ch = _read_byte(0.05)
            if ch is None or ('\x40' <= ch <= '\x7e'):
                break
    elif ch == 'O':
        # SS3 sequence (e.g. arrow keys in some terminals): one more byte
        _read_byte(0.05)
    # else: two-char escape sequence, already consumed


def _monitor_loop():
    """Background thread that watches for double-escape."""
    last_esc_time = None

    while True:
        _monitor_active.wait()

        ch = _read_byte(0.1)
        if ch is None:
            continue

        if ch == '\x1b':
            # Check if this is the start of an ANSI sequence
            next_ch = _read_byte(0.05)
            if next_ch is not None:
                # ANSI sequence — consume it and ignore
                if next_ch == '[':
                    while True:
                        c = _read_byte(0.05)
                        if c is None or ('\x40' <= c <= '\x7e'):
                            break
                elif next_ch == 'O':
                    _read_byte(0.05)
                # else: two-char sequence, consumed
                continue

            # Bare escape — check for double-escape
            now = time.monotonic()
            if last_esc_time is not None and (now - last_esc_time) < 0.4:
                _cancel_event.set()
                last_esc_time = None
            else:
                last_esc_time = now
        else:
            # Non-escape keypress — consume it (prevent buffer leakage)
            last_esc_time = None


# Start the monitor thread as a daemon
_monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
_monitor_thread.start()


class cancellable:
    """Context manager that enables double-escape cancellation.

    Combines reset + cbreak mode + monitor activation.
    Gracefully degrades when stdin isn't a terminal.
    """

    def __init__(self):
        self._cbreak = None

    def __enter__(self):
        reset()
        if _tui_mode:
            return self  # prompt_toolkit manages the terminal
        if sys.stdin.isatty():
            self._cbreak = cbreak_mode()
            self._cbreak.__enter__()
            _monitor_active.set()
        return self

    def __exit__(self, *exc):
        _monitor_active.clear()
        if self._cbreak is not None:
            self._cbreak.__exit__(*exc)
            self._cbreak = None
        return False


def _restore_terminal():
    """atexit handler — restore terminal settings if we crashed in cbreak mode."""
    global _original_termios
    if _original_termios is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_termios)
        except Exception:
            pass


atexit.register(_restore_terminal)
