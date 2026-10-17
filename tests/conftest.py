"""Shared fixtures for the in-process gateway."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from llm_guard.api.app import create_app
from llm_guard.config import Settings, load_settings

REGULATED_KEY = "lg_test_regulated_demo_key_0001"
LATENCY_KEY = "lg_test_latency_demo_key_0001"
ADMIN_KEY = "lg_admin_local_dev"


def isolated_settings(tmp_path: Path) -> Settings:
    settings = load_settings()
    server = settings.server.model_copy(update={"audit_log_path": str(tmp_path / "audit.jsonl")})
    return settings.model_copy(update={"server": server})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return isolated_settings(tmp_path)


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[tuple[httpx.AsyncClient, object]]:
    application = create_app(settings)
    transport = ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://llm-guard.test") as http:
        yield http, application


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}
