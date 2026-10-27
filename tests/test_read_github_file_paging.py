"""read_github_file pages large files past the host's tool-result cap (#11).

The host replaces a tool result whose JSON encoding exceeds
``KESTREL_MAX_TOOL_RESULT_CHARS`` with a head-and-tail preview. These tests
read files through the tool page by page, on both the network and the cache
path, and check every page fits that cap as the host measures it.
"""

import json
from unittest.mock import patch

import pytest

from kestrel_sdk.tools.result import ToolResult

from kestrel_feature_github.client import GitHubClient, GitHubClientError
from kestrel_feature_github.feature import GitHubFeature
from kestrel_feature_github.models import FileContent

TOOL = "read_github_file"
REPO = "KestrelSovereignAI/kestrel-sovereign"
PATH = "big.txt"
CAP = 8000

# ASCII, characters JSON escapes, BMP non-ASCII, and astral characters (each a
# twelve-character surrogate pair once serialized).
_MIXED_ALPHABET = 'ab c"\\\n\téü日本🦅😀𝕏'
_EMOJI_ALPHABET = "🦅😀🎉🚀🧪"


def _cycle(alphabet: str, n: int) -> str:
    return "".join(alphabet[i % len(alphabet)] for i in range(n))


MIXED_50K = _cycle(_MIXED_ALPHABET, 50_000)
EMOJI_50K = _cycle(_EMOJI_ALPHABET, 50_000)


def _host_measure(result: ToolResult) -> int:
    """How the host measures a result against the cap: json.dumps defaults
    over to_dict() plus the tool wrapper's ``tool`` and ``success`` keys."""
    return len(json.dumps({**result.to_dict(), "tool": TOOL, "success": True}))


def _file(text: str) -> FileContent:
    return FileContent(
        path=PATH,
        content=text,
        sha="blobsha",
        size=len(text.encode("utf-8")),
        repo=REPO,
        ref="main",
    )


@pytest.fixture
def feature(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_PAT", "test_token")
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))
    monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", str(CAP))
    return GitHubFeature()


async def _read_all_pages(feature, cap=CAP, **kwargs):
    """Follow next_offset from 0 to the end, checking each page as it comes."""
    pages = []
    offset = 0
    while offset is not None:
        result = await feature.read_github_file(
            repo="self", path=PATH, offset=offset, **kwargs
        )
        assert result.error is None, result.error
        assert _host_measure(result) <= cap
        data = result.data
        assert data["offset"] == offset
        assert data["returned_chars"] == len(data["content"])
        if data["next_offset"] is not None:
            assert data["next_offset"] == offset + data["returned_chars"]
            assert data["returned_chars"] > 0
        pages.append(data)
        offset = data["next_offset"]
        assert len(pages) <= 1000, "paging made no progress"
    return pages


class TestReassembly:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [MIXED_50K, EMOJI_50K], ids=["mixed", "all-emoji"])
    async def test_50k_file_read_page_by_page_reassembles_exactly(self, feature, text):
        assert len(text) == 50_000
        with patch.object(
            GitHubClient, "get_file_content", return_value=_file(text)
        ) as fetch:
            pages = await _read_all_pages(feature)

        assert "".join(p["content"] for p in pages) == text
        assert len(pages) > 1
        assert all(p["total_chars"] == 50_000 for p in pages)
        assert pages[-1]["next_offset"] is None
        # The first page came from the network, every later one from the cache.
        fetch.assert_awaited_once_with(REPO, PATH, "main")
        assert pages[0]["cached"] is False
        assert all(p["cached"] is True for p in pages[1:])

    @pytest.mark.asyncio
    async def test_pages_fit_a_smaller_configured_cap(self, feature, monkeypatch):
        monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", "2000")
        with patch.object(GitHubClient, "get_file_content", return_value=_file(MIXED_50K)):
            pages = await _read_all_pages(feature, cap=2000)

        assert "".join(p["content"] for p in pages) == MIXED_50K

    @pytest.mark.asyncio
    async def test_explicit_limit_pages_reassemble(self, feature):
        with patch.object(GitHubClient, "get_file_content", return_value=_file(MIXED_50K)):
            pages = await _read_all_pages(feature, limit=300)

        assert "".join(p["content"] for p in pages) == MIXED_50K
        assert all(p["returned_chars"] <= 300 for p in pages)


class TestOffsetAndLimit:
    @pytest.mark.asyncio
    async def test_call_without_paging_args_returns_small_file_whole(self, feature):
        with patch.object(GitHubClient, "get_file_content", return_value=_file("print('hi')\n")):
            result = await feature.read_github_file(repo="self", path=PATH)

        assert result.error is None
        assert result.data["content"] == "print('hi')\n"
        assert result.data["offset"] == 0
        assert result.data["returned_chars"] == 12
        assert result.data["total_chars"] == 12
        assert result.data["next_offset"] is None
        assert result.data["repo"] == REPO
        assert result.data["path"] == PATH
        assert result.data["ref"] == "main"
        assert result.data["sha"] == "blobsha"

    @pytest.mark.asyncio
    async def test_call_without_paging_args_returns_first_chunk_of_large_file(self, feature):
        with patch.object(GitHubClient, "get_file_content", return_value=_file(MIXED_50K)):
            result = await feature.read_github_file(repo="self", path=PATH)

        assert result.error is None
        assert _host_measure(result) <= CAP
        assert MIXED_50K.startswith(result.data["content"])
        assert result.data["next_offset"] == result.data["returned_chars"]
        assert f"offset={result.data['next_offset']}" in result.confirmation

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offset", [50_000, 50_001, 10**9])
    async def test_out_of_range_offset_returns_empty_chunk(self, feature, offset):
        with patch.object(GitHubClient, "get_file_content", return_value=_file(MIXED_50K)):
            # Network path, then cache path.
            for _ in range(2):
                result = await feature.read_github_file(
                    repo="self", path=PATH, offset=offset
                )
                assert result.error is None, result.error
                assert result.data["content"] == ""
                assert result.data["returned_chars"] == 0
                assert result.data["total_chars"] == 50_000
                assert result.data["next_offset"] is None

    @pytest.mark.asyncio
    async def test_network_path_applies_offset_and_limit(self, feature):
        with patch.object(
            GitHubClient, "get_file_content", return_value=_file(MIXED_50K)
        ) as fetch:
            result = await feature.read_github_file(
                repo="self", path=PATH, offset=100, limit=50
            )

        fetch.assert_awaited_once()
        assert result.data["cached"] is False
        assert result.data["content"] == MIXED_50K[100:150]
        assert result.data["offset"] == 100
        assert result.data["next_offset"] == 150

    @pytest.mark.asyncio
    async def test_cache_path_applies_offset_and_limit(self, feature):
        await feature.cache.set(_file(MIXED_50K))

        with patch.object(GitHubClient, "get_file_content") as fetch:
            result = await feature.read_github_file(
                repo="self", path=PATH, offset=49_990, limit=50
            )

        fetch.assert_not_awaited()
        assert result.data["cached"] is True
        assert result.data["content"] == MIXED_50K[49_990:]
        assert result.data["returned_chars"] == 10
        assert result.data["next_offset"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"offset": -1}, "offset must be at least 0"),
            ({"limit": 0}, "limit must be at least 1"),
        ],
    )
    async def test_invalid_paging_args_fail_without_fetching(self, feature, kwargs, message):
        with patch.object(GitHubClient, "get_file_content") as fetch:
            result = await feature.read_github_file(repo="self", path=PATH, **kwargs)

        fetch.assert_not_awaited()
        assert result.error is not None
        assert message in result.error

    @pytest.mark.asyncio
    async def test_client_error_is_reported(self, feature):
        with patch.object(
            GitHubClient,
            "get_file_content",
            side_effect=GitHubClientError("File not found: big.txt in x/y", 404),
        ):
            result = await feature.read_github_file(repo="self", path=PATH, offset=10)

        assert result.error is not None
        assert "Error reading big.txt" in result.error


def _registered_tool(feature):
    return {t.name: t for t in feature.get_tools()}[TOOL]


class TestRegisteredTool:
    """The tool as the host sees it: the schema the SDK advertises to the
    model, and the SDK's wrapper calling it with schema-conforming arguments."""

    def test_schema_advertises_offset_and_limit_as_optional_integers(self, feature):
        params = {p.name: p for p in _registered_tool(feature).schema.parameters}
        for name in ("offset", "limit"):
            assert params[name].type == "integer", (name, params[name].type)
            assert params[name].required is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [50, None], ids=["integer", "null"])
    async def test_registered_tool_accepts_integer_or_null_limit(self, feature, limit):
        with patch.object(GitHubClient, "get_file_content", return_value=_file(MIXED_50K)):
            payload = await _registered_tool(feature).execute(
                repo="self", path=PATH, offset=100, limit=limit
            )

        assert payload.get("status") == "ok", payload
        data = payload["data"]
        assert data["offset"] == 100
        assert data["content"] == MIXED_50K[100 : data["next_offset"]]
        if limit is not None:
            assert data["next_offset"] == 150

    @pytest.mark.asyncio
    async def test_registered_tool_pages_reassemble_within_cap(self, feature):
        tool = _registered_tool(feature)
        chunks = []
        offset = 0
        with patch.object(GitHubClient, "get_file_content", return_value=_file(EMOJI_50K)):
            while offset is not None:
                payload = await tool.execute(repo="self", path=PATH, offset=offset)
                assert payload.get("status") == "ok", payload
                # The wrapper's own envelope, measured the way the host does.
                assert len(json.dumps(payload)) <= CAP
                chunks.append(payload["data"]["content"])
                offset = payload["data"]["next_offset"]
                assert len(chunks) <= 1000, "paging made no progress"

        assert "".join(chunks) == EMOJI_50K
