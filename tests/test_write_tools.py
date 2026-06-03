"""Wire-level tests for the GitHubFeature write surface (#1502).

Each test patches ``client._get_client`` to return an ``AsyncMock`` whose
HTTP-verb methods (``post`` / ``patch`` / ``put`` / ``delete``) return a
``MockResponse`` shaped like ``httpx.Response``. This validates:

  - the URL the client hits
  - the JSON payload it sends
  - the status-code dispatch path (success vs structured error)
  - the ToolResult envelope at the @tool layer

No real network. No real GitHub. The whole point of write tools is that
they have side effects, so the unit boundary is "did we form the right
request"; integration is left to a smoke test in the host repo.
"""
from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_feature_github.client import GitHubClient, GitHubClientError
from kestrel_feature_github.feature import GitHubFeature


class _MockResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or (str(payload) if payload is not None else "")

    def json(self) -> Any:
        return self._payload


def _http_mock(
    *,
    post: Optional[_MockResponse] = None,
    patch_: Optional[_MockResponse] = None,
    put: Optional[_MockResponse] = None,
    delete: Optional[_MockResponse] = None,
):
    """Build an AsyncMock httpx-like client with the given verb responses."""
    m = AsyncMock()
    if post is not None:
        m.post = AsyncMock(return_value=post)
    if patch_ is not None:
        m.patch = AsyncMock(return_value=patch_)
    if put is not None:
        m.put = AsyncMock(return_value=put)
    if delete is not None:
        m.delete = AsyncMock(return_value=delete)
    return m


@pytest.fixture
def feature():
    with patch.dict("os.environ", {"GITHUB_PAT": "test_token"}):
        return GitHubFeature()


# --------------------------------------------------------------------- #
# create_github_issue                                                    #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_create_github_issue_success(feature):
    http = _http_mock(post=_MockResponse(
        201,
        {"number": 99, "html_url": "https://github.com/x/y/issues/99"},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.create_github_issue(
            title="Hello", body="World",
            repo="x/y", labels="bug, talon", assignees="alice",
        )
    assert result.error is None, result.error
    assert result.data["number"] == 99
    assert result.data["url"].endswith("/issues/99")
    # Wire check: payload includes parsed labels/assignees lists
    call = http.post.call_args
    assert call.args[0] == "/repos/x/y/issues"
    assert call.kwargs["json"] == {
        "title": "Hello",
        "body": "World",
        "labels": ["bug", "talon"],
        "assignees": ["alice"],
    }


@pytest.mark.asyncio
async def test_create_github_issue_filters_empty_csv_entries(feature):
    """Trailing/extra commas in labels or assignees must not send empty
    strings to GitHub (422 from validation). Codex round 1 P2."""
    http = _http_mock(post=_MockResponse(
        201,
        {"number": 100, "html_url": "https://github.com/x/y/issues/100"},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.create_github_issue(
            title="t", body="b", repo="x/y",
            labels="bug, ,talon,", assignees="alice, , bob,",
        )
    assert result.error is None, result.error
    payload = http.post.call_args.kwargs["json"]
    assert payload["labels"] == ["bug", "talon"]  # empties dropped
    assert payload["assignees"] == ["alice", "bob"]  # empties dropped


@pytest.mark.asyncio
async def test_create_github_issue_404(feature):
    http = _http_mock(post=_MockResponse(404, text="Not found"))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.create_github_issue(
            title="t", body="b", repo="x/missing",
        )
    assert result.error is not None
    assert "Repository not found" in result.error


# --------------------------------------------------------------------- #
# add_github_issue_comment                                               #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_add_github_issue_comment_success(feature):
    http = _http_mock(post=_MockResponse(
        201,
        {"id": 4242, "html_url": "https://github.com/x/y/issues/1#issuecomment-4242"},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.add_github_issue_comment(
            issue_number=1, body="Looks good", repo="x/y",
        )
    assert result.error is None, result.error
    assert result.data["comment_id"] == 4242
    call = http.post.call_args
    assert call.args[0] == "/repos/x/y/issues/1/comments"
    assert call.kwargs["json"] == {"body": "Looks good"}


# --------------------------------------------------------------------- #
# add_github_label                                                       #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_add_github_label_parses_csv(feature):
    http = _http_mock(post=_MockResponse(
        200,
        [{"name": "bug"}, {"name": "talon"}, {"name": "agent-claimed"}],
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.add_github_label(
            issue_number=7, labels="bug, talon , agent-claimed", repo="x/y",
        )
    assert result.error is None, result.error
    assert result.data["added"] == ["bug", "talon", "agent-claimed"]
    assert result.data["current"] == ["bug", "talon", "agent-claimed"]
    call = http.post.call_args
    assert call.args[0] == "/repos/x/y/issues/7/labels"
    assert call.kwargs["json"] == {"labels": ["bug", "talon", "agent-claimed"]}


@pytest.mark.asyncio
async def test_add_github_label_empty_input(feature):
    # No HTTP should fire if input is empty after stripping.
    http = _http_mock(post=_MockResponse(500))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.add_github_label(
            issue_number=7, labels=" , , ", repo="x/y",
        )
    assert result.error is not None
    assert "No labels" in result.error
    http.post.assert_not_called()


# --------------------------------------------------------------------- #
# remove_github_label                                                    #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_remove_github_label_204_is_success(feature):
    """The happy path returns 204 with no body."""
    http = _http_mock(delete=_MockResponse(204))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.remove_github_label(
            issue_number=42, label="agent-failed", repo="x/y",
        )
    assert result.error is None, result.error
    call = http.delete.call_args
    # Label is URL-encoded (the helper escapes safe='')
    assert "agent-failed" in call.args[0]


@pytest.mark.asyncio
async def test_remove_github_label_404_label_not_on_issue_idempotent(feature):
    """404 with 'Label does not exist' body is the documented label-not-on-
    issue response — idempotent success. Codex round 1 P2."""
    http = _http_mock(delete=_MockResponse(
        404,
        text='{"message":"Label does not exist","documentation_url":"..."}',
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.remove_github_label(
            issue_number=42, label="agent-failed", repo="x/y",
        )
    assert result.error is None, result.error


@pytest.mark.asyncio
async def test_remove_github_label_404_issue_missing_fails(feature):
    """404 with any other body (issue missing, repo missing, no access) is a
    REAL failure — don't tell the agent it removed a label from an issue
    that doesn't exist. Codex round 1 P2."""
    http = _http_mock(delete=_MockResponse(
        404, text='{"message":"Not Found"}',
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.remove_github_label(
            issue_number=99999, label="agent-failed", repo="x/missing",
        )
    assert result.error is not None
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_remove_github_label_url_encodes_slashes(feature):
    http = _http_mock(delete=_MockResponse(204))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.remove_github_label(
            issue_number=42, label="area/memory", repo="x/y",
        )
    assert result.error is None, result.error
    call = http.delete.call_args
    # The "/" in the label is URL-encoded as %2F so it doesn't break the path.
    assert "area%2Fmemory" in call.args[0]


# --------------------------------------------------------------------- #
# close_github_issue / reopen_github_issue                               #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_close_github_issue_default_completed(feature):
    http = _http_mock(patch_=_MockResponse(
        200,
        {"state": "closed", "state_reason": "completed"},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.close_github_issue(
            issue_number=11, repo="x/y",
        )
    assert result.error is None, result.error
    assert result.data["state"] == "closed"
    assert result.data["state_reason"] == "completed"
    call = http.patch.call_args
    assert call.args[0] == "/repos/x/y/issues/11"
    assert call.kwargs["json"] == {"state": "closed", "state_reason": "completed"}


@pytest.mark.asyncio
async def test_close_github_issue_not_planned(feature):
    http = _http_mock(patch_=_MockResponse(
        200,
        {"state": "closed", "state_reason": "not_planned"},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.close_github_issue(
            issue_number=12, repo="x/y", state_reason="not_planned",
        )
    assert result.error is None, result.error
    assert result.data["state_reason"] == "not_planned"


@pytest.mark.asyncio
async def test_reopen_github_issue(feature):
    http = _http_mock(patch_=_MockResponse(
        200,
        {"state": "open"},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.reopen_github_issue(
            issue_number=13, repo="x/y",
        )
    assert result.error is None, result.error
    assert result.data["state"] == "open"
    call = http.patch.call_args
    assert call.kwargs["json"] == {"state": "open", "state_reason": "reopened"}


# --------------------------------------------------------------------- #
# create_github_pull_request                                             #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_create_pr_success(feature):
    http = _http_mock(post=_MockResponse(
        201,
        {
            "number": 501,
            "html_url": "https://github.com/x/y/pull/501",
            "head": {"sha": "abc1234567890"},
            "draft": False,
        },
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.create_github_pull_request(
            title="fix things",
            head="feature/x",
            base="main",
            body="closes #500",
            repo="x/y",
        )
    assert result.error is None, result.error
    assert result.data["number"] == 501
    assert result.data["url"].endswith("/pull/501")
    assert result.data["head_sha"] == "abc1234567890"
    call = http.post.call_args
    assert call.args[0] == "/repos/x/y/pulls"
    payload = call.kwargs["json"]
    assert payload["title"] == "fix things"
    assert payload["head"] == "feature/x"
    assert payload["base"] == "main"
    assert payload["body"] == "closes #500"
    assert payload["draft"] is False


@pytest.mark.asyncio
async def test_create_pr_422_no_diff(feature):
    http = _http_mock(post=_MockResponse(422, text="No commits between main and feature/x"))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.create_github_pull_request(
            title="x", head="feature/x", base="main", repo="x/y",
        )
    assert result.error is not None
    assert "validation failed" in result.error.lower()


# --------------------------------------------------------------------- #
# merge_github_pull_request                                              #
# --------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_merge_pr_default_squash(feature):
    http = _http_mock(put=_MockResponse(
        200,
        {"sha": "deadbeef" * 5, "merged": True},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.merge_github_pull_request(
            pull_number=501, repo="x/y",
        )
    assert result.error is None, result.error
    assert result.data["merge_method"] == "squash"
    assert result.data["merged"] is True
    call = http.put.call_args
    assert call.args[0] == "/repos/x/y/pulls/501/merge"
    assert call.kwargs["json"] == {"merge_method": "squash"}


@pytest.mark.asyncio
async def test_merge_pr_with_sha_pin(feature):
    """Passing sha ensures the merge fails-closed if HEAD has moved."""
    http = _http_mock(put=_MockResponse(
        200,
        {"sha": "0" * 40, "merged": True},
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.merge_github_pull_request(
            pull_number=501, repo="x/y", sha="abc1234567890",
            commit_title="Custom title", commit_message="Body",
        )
    assert result.error is None, result.error
    payload = http.put.call_args.kwargs["json"]
    assert payload["sha"] == "abc1234567890"
    assert payload["commit_title"] == "Custom title"
    assert payload["commit_message"] == "Body"


@pytest.mark.asyncio
async def test_merge_pr_405_not_mergeable(feature):
    http = _http_mock(put=_MockResponse(
        405, text="Pull Request is not mergeable",
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.merge_github_pull_request(
            pull_number=501, repo="x/y",
        )
    assert result.error is not None
    assert "not mergeable" in result.error.lower()


@pytest.mark.asyncio
async def test_merge_pr_409_sha_mismatch(feature):
    """When sha is pinned and HEAD moved, GitHub returns 409 — surface it cleanly."""
    http = _http_mock(put=_MockResponse(
        409, text="Head branch was modified",
    ))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.merge_github_pull_request(
            pull_number=501, repo="x/y", sha="abc1234",
        )
    assert result.error is not None
    assert "head sha mismatch" in result.error.lower()


@pytest.mark.asyncio
async def test_merge_pr_invalid_method_rejected_pre_flight(feature):
    """merge_method validation happens before any HTTP fires."""
    http = _http_mock(put=_MockResponse(500))
    with patch.object(GitHubClient, "_get_client", return_value=http):
        result = await feature.merge_github_pull_request(
            pull_number=501, repo="x/y", merge_method="cherry-pick",
        )
    assert result.error is not None
    assert "merge_method must be" in result.error
    http.put.assert_not_called()


# --------------------------------------------------------------------- #
# Categories — write tools are SYSTEM category, not DATA_ACCESS          #
# --------------------------------------------------------------------- #

def test_write_tools_use_system_category():
    """Write tools must declare SYSTEM category so the host's permission
    layer can apply distinct policy from data-access reads."""
    from kestrel_sdk.tools.base import ToolCategory
    feature = GitHubFeature()
    tools_by_name = {t.name: t for t in feature.get_tools()}
    for name in (
        "create_github_issue",
        "add_github_issue_comment",
        "add_github_label",
        "remove_github_label",
        "close_github_issue",
        "reopen_github_issue",
        "create_github_pull_request",
        "merge_github_pull_request",
    ):
        assert tools_by_name[name].schema.category == ToolCategory.SYSTEM, (
            f"{name} must be SYSTEM (write surface), got "
            f"{tools_by_name[name].schema.category}"
        )
