"""Cross-origin access to the API.

The API has no authentication, so a browser should only let the VectorVault UI
read its responses. A page on any other site gets no CORS allow header, and the
browser then withholds the response from that page's script.
"""

from __future__ import annotations

import pytest

UI_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
FOREIGN_ORIGIN = "https://example.com"

ALLOW_ORIGIN = "access-control-allow-origin"
ALLOW_CREDENTIALS = "access-control-allow-credentials"


def preflight(client, origin: str):
    return client.options(
        "/search",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


@pytest.mark.parametrize("origin", UI_ORIGINS)
def test_ui_origin_is_allowed(client, origin):
    r = client.get("/health", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers[ALLOW_ORIGIN] == origin

    r = preflight(client, origin)
    assert r.status_code == 200
    assert r.headers[ALLOW_ORIGIN] == origin


def test_foreign_origin_gets_no_allow_header(client):
    r = client.get("/health", headers={"Origin": FOREIGN_ORIGIN})
    assert ALLOW_ORIGIN not in r.headers


def test_foreign_origin_preflight_is_refused(client):
    r = preflight(client, FOREIGN_ORIGIN)
    assert r.status_code == 400
    assert ALLOW_ORIGIN not in r.headers


def test_credentials_are_not_allowed(client):
    for r in (
        client.get("/health", headers={"Origin": UI_ORIGINS[0]}),
        preflight(client, UI_ORIGINS[0]),
    ):
        assert ALLOW_CREDENTIALS not in r.headers


def test_default_origins_are_the_ui(app_module, monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert app_module.cors_origins() == UI_ORIGINS


def test_cors_origins_env_var_replaces_the_default(app_module, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", " https://search.example.org ,http://localhost:3000,")
    assert app_module.cors_origins() == ["https://search.example.org", "http://localhost:3000"]


def test_empty_cors_origins_allows_no_origin(app_module, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "")
    assert app_module.cors_origins() == []
