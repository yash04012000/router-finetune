"""Tests for logging setup: file gets DEBUG, console level follows --verbose, tail reads the end."""

import logging

from router import log


def fresh_logger():
    root = logging.getLogger("router")
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    return root


def test_file_gets_debug_but_console_only_info(tmp_path):
    root = fresh_logger()
    try:
        log.setup_logging(verbose=False, log_file=tmp_path / "x.log")
        logging.getLogger("router.demo").debug("detail line")
        logging.getLogger("router.demo").info("normal line")
        console, file_handler = root.handlers
        assert console.level == logging.INFO and file_handler.level == logging.DEBUG
        text = (tmp_path / "x.log").read_text(encoding="utf-8")
        assert "detail line" in text and "normal line" in text and "router.demo" in text
    finally:
        fresh_logger()


def test_verbose_lowers_console_level_and_setup_is_idempotent(tmp_path):
    root = fresh_logger()
    try:
        log.setup_logging(verbose=False, log_file=tmp_path / "x.log")
        log.setup_logging(verbose=True, log_file=tmp_path / "x.log")
        assert len(root.handlers) == 2  # not duplicated
        assert root.handlers[0].level == logging.DEBUG
    finally:
        fresh_logger()


def test_tail_returns_last_lines_and_handles_missing_file(tmp_path):
    f = tmp_path / "t.log"
    assert log.tail(5, f) == []
    f.write_text("\n".join(f"line {i}" for i in range(10)), encoding="utf-8")
    assert log.tail(3, f) == ["line 7", "line 8", "line 9"]
