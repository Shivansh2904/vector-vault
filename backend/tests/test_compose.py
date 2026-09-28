"""Checks on docker-compose.yml.

The frontend waits for the backend's healthcheck, so a healthcheck that can
never pass means the frontend never starts.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def healthcheck(compose) -> list[str]:
    test = compose["services"]["backend"]["healthcheck"]["test"]
    assert test[0] == "CMD"
    return test[1:]


def test_ports_are_published_on_localhost_only(compose):
    # The API has no authentication, so it must not be reachable from the LAN.
    for name, service in compose["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} on every interface"


def test_healthcheck_only_needs_what_the_backend_image_has(healthcheck):
    base = (REPO / "backend" / "Dockerfile").read_text(encoding="utf-8").split()[1]
    assert base.startswith("python:") and base.endswith("-slim"), base
    # The slim Python images ship python but not curl or wget.
    assert healthcheck[0] in {"python", "python3"}, (
        f"{healthcheck[0]!r} is not installed in {base}, so the backend never "
        f"reports healthy and the frontend never starts"
    )


@pytest.mark.parametrize("status, healthy", [(200, True), (503, False)])
def test_healthcheck_fails_when_health_does_not_return_ok(healthcheck, status, healthy):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code = healthcheck[-1]
        assert "localhost:8000" in code
        code = code.replace("localhost:8000", f"127.0.0.1:{server.server_port}")
        argv = [sys.executable, *healthcheck[1:-1], code]
        result = subprocess.run(argv, capture_output=True, timeout=30)
    finally:
        server.shutdown()
    assert (result.returncode == 0) == healthy, result.stderr.decode()
