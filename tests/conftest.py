"""Shared test fixtures."""

import threading
from http.server import ThreadingHTTPServer

import pytest
from scripts import serve


@pytest.fixture(scope="session")
def base_url():
    """A real playground server on a free local port, for the whole test run."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)  # port 0 = pick any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
