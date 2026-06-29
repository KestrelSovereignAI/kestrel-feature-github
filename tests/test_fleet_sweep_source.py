"""Tests for fleet resolution + the fleet_stalled_sweep signal source (#1523 2b)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_feature_github.feature import GitHubFeature, GITHUB_SELF_REPO
from kestrel_feature_github.stalled_sweep_source import (
    SOURCE_NAME,
    _schema,
    build_fleet_stalled_sweep_registration,
)

NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)


def _iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


class _FakeRegistry:
    def __init__(self):
        self.sources = {}

    def get(self, name):
        return self.sources.get(name)

    def register(self, reg):
        self.sources[reg.name] = reg


def _feature(agent=None):
    with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
        return GitHubFeature(agent=agent)


class TestFleetResolution:
    def test_fleet_defaults_to_self_when_unset(self):
        with patch.dict("os.environ", {"GITHUB_PAT": "t"}, clear=False):
            with patch.dict("os.environ", {"GITHUB_FLEET_REPOS": ""}):
                assert _feature()._resolve_repos("fleet") == [GITHUB_SELF_REPO]

    def test_fleet_uses_configured_env(self):
        with patch.dict("os.environ", {"GITHUB_FLEET_REPOS": "o/a, o/b ,o/c"}):
            assert _feature()._resolve_repos("fleet") == ["o/a", "o/b", "o/c"]

    def test_self_alias_and_explicit_list(self):
        f = _feature()
        assert f._resolve_repos("self") == [GITHUB_SELF_REPO]
        assert f._resolve_repos("o/a, o/b") == ["o/a", "o/b"]


class TestSourceSchema:
    def test_accepts_known_keys(self):
        assert _schema({"stale_days": 5, "repos": "fleet"}) == {
            "stale_days": 5,
            "repos": "fleet",
        }
        assert _schema({}) == {}

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError):
            _schema({"nope": 1})

    def test_rejects_bad_stale_days(self):
        with pytest.raises(ValueError):
            _schema({"stale_days": 0})
        with pytest.raises(ValueError):
            _schema({"stale_days": "soon"})

    def test_registration_is_action_with_handler(self):
        async def _h(payload):
            return {}

        reg = build_fleet_stalled_sweep_registration(_h)
        assert reg.name == SOURCE_NAME
        assert reg.handler is _h


class TestSourceRegistration:
    @pytest.mark.asyncio
    async def test_initialize_registers_source(self):
        registry = _FakeRegistry()
        agent = SimpleNamespace(signal_registry=registry)
        feature = _feature(agent=agent)

        await feature.initialize()

        assert registry.get(SOURCE_NAME) is not None

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self):
        registry = _FakeRegistry()
        feature = _feature(agent=SimpleNamespace(signal_registry=registry))
        await feature.initialize()
        first = registry.get(SOURCE_NAME)
        await feature.initialize()
        assert registry.get(SOURCE_NAME) is first  # not re-registered

    @pytest.mark.asyncio
    async def test_initialize_without_registry_is_noop(self):
        # agent has no signal_registry — must not raise
        feature = _feature(agent=SimpleNamespace())
        await feature.initialize()

    @pytest.mark.asyncio
    async def test_sweep_handler_returns_findings(self):
        feature = _feature(agent=SimpleNamespace(signal_registry=_FakeRegistry()))
        feature.client.get_repo_info = AsyncMock(return_value={"default_branch": "main"})
        feature.client.list_issues = AsyncMock(
            return_value=[
                {
                    "number": 1,
                    "title": "t",
                    "html_url": "u",
                    "labels": [{"name": "agent-claimed"}],
                    "updated_at": _iso(5),
                }
            ]
        )
        feature.client.list_pull_requests = AsyncMock(return_value=[])
        feature.client.list_workflow_runs = AsyncMock(return_value=[])

        result = await feature._fleet_sweep_handler({"stale_days": 3, "repos": "o/r"})

        assert result["repos_scanned"] == ["o/r"]
        assert [f["kind"] for f in result["findings"]] == ["stalled_claim"]
