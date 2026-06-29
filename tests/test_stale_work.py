"""Tests for stalled-work detection (sovereign #1523)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_feature_github.feature import GitHubFeature
from kestrel_feature_github.stale_work import classify_stale_work

NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=timezone.utc)


def _iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _issue(number, *, labels, days_ago, title="t"):
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/o/r/issues/{number}",
        "labels": [{"name": n} for n in labels],
        "updated_at": _iso(days_ago),
    }


def _pr(number, *, days_ago, draft=False, title="t"):
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/o/r/pull/{number}",
        "draft": draft,
        "updated_at": _iso(days_ago),
    }


class TestClassify:
    def test_red_default_branch_is_high(self):
        items = classify_stale_work(
            "o/r",
            issues=[],
            pull_requests=[],
            latest_default_run={"conclusion": "failure", "html_url": "u", "display_title": "CI"},
            default_branch="main",
            now=NOW,
        )
        assert len(items) == 1
        assert items[0].kind == "red_default_branch"
        assert items[0].severity == "high"

    def test_green_default_branch_yields_nothing(self):
        items = classify_stale_work(
            "o/r",
            issues=[],
            pull_requests=[],
            latest_default_run={"conclusion": "success"},
            default_branch="main",
            now=NOW,
        )
        assert items == []

    def test_stalled_claim_detected_only_when_claimed_and_stale(self):
        issues = [
            _issue(1, labels=["agent-claimed"], days_ago=5),   # stalled claim
            _issue(2, labels=["agent-claimed"], days_ago=1),   # claimed but fresh
            _issue(3, labels=["enhancement"], days_ago=30),    # stale but not claimed
        ]
        items = classify_stale_work(
            "o/r", issues=issues, pull_requests=[], latest_default_run=None,
            default_branch="main", now=NOW, stale_days=3,
        )
        assert [it.ref for it in items] == ["#1"]
        assert items[0].kind == "stalled_claim"
        assert items[0].age_days == 5

    def test_stale_pr_and_draft_distinguished(self):
        prs = [
            _pr(10, days_ago=4, draft=False),
            _pr(11, days_ago=10, draft=True),
            _pr(12, days_ago=1),  # fresh, excluded
        ]
        items = classify_stale_work(
            "o/r", issues=[], pull_requests=prs, latest_default_run=None,
            default_branch="main", now=NOW, stale_days=3,
        )
        kinds = {it.ref: it.kind for it in items}
        assert kinds == {"#10": "stale_pr", "#11": "stale_draft"}


class TestScanStaleWorkTool:
    @pytest.fixture
    def feature(self):
        with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
            f = GitHubFeature()
            _ = f.cache
            return f

    def test_tool_is_registered(self, feature):
        with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
            assert "scan_stale_work" in [t.name for t in feature.get_tools()]

    @pytest.mark.asyncio
    async def test_scan_aggregates_and_sorts_by_severity(self, feature):
        feature.client.get_repo_info = AsyncMock(return_value={"default_branch": "main"})
        feature.client.list_issues = AsyncMock(
            return_value=[_issue(1, labels=["agent-claimed"], days_ago=5)]
        )
        feature.client.list_pull_requests = AsyncMock(
            return_value=[_pr(10, days_ago=4)]
        )
        feature.client.list_workflow_runs = AsyncMock(
            return_value=[{"conclusion": "failure", "html_url": "u", "display_title": "CI"}]
        )

        result = await feature.scan_stale_work(repos="o/r", stale_days=3)

        assert result.status.value == "ok"
        findings = result.data["findings"]
        assert len(findings) == 3
        # high-severity (red branch + stalled claim) sort before the medium PR
        assert findings[-1]["kind"] == "stale_pr"
        assert {f["kind"] for f in findings} == {
            "red_default_branch", "stalled_claim", "stale_pr",
        }

    @pytest.mark.asyncio
    async def test_scan_records_per_repo_errors_without_aborting(self, feature):
        from kestrel_feature_github.client import GitHubClientError

        feature.client.get_repo_info = AsyncMock(side_effect=GitHubClientError("boom", 404))

        result = await feature.scan_stale_work(repos="o/missing", stale_days=3)

        assert result.status.value == "ok"
        assert result.data["findings"] == []
        assert result.data["errors"][0]["repo"] == "o/missing"
