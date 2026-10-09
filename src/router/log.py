"""Logging for the whole project, set up in one place.

Every module does `logger = logging.getLogger("router.<something>")` and logs normally.
A script (or the server) calls `setup_logging()` once at the start, which sends those messages to:

    the console           INFO and above   (DEBUG too if you pass --verbose)
    logs/router.log       EVERYTHING, DEBUG and above, rotated at 1 MB (3 old files kept)

So when something looks wrong, open logs/router.log: it has more detail than the console.
"""

import logging
import logging.handlers
from pathlib import Path

from router.intents import REPO_ROOT

LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "router.log"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(verbose: bool = False, log_file: Path = LOG_FILE) -> logging.Logger:
    root = logging.getLogger("router")
    console_level = logging.DEBUG if verbose else logging.INFO
    if root.handlers:  # already set up (e.g. called twice): only update the console level
        for handler in root.handlers:
            if not isinstance(handler, logging.FileHandler):
                handler.setLevel(console_level)
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False  # we handle output ourselves, don't double-print via the root logger
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return root


def tail(n: int = 200, log_file: Path = LOG_FILE) -> list[str]:
    """The last n lines of the log file (empty list if there is no log yet)."""
    if not log_file.exists():
        return []
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]
