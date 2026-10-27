"""Tests for the ranged-read helper behind read_github_file paging (#11)."""

import json

import pytest

from kestrel_sdk.tools.result import ToolResult

from kestrel_feature_github.paging import (
    PageRequestError,
    paged_text_result,
    serialized_result_len,
    tool_result_cap,
    validate_page,
)

TOOL = "read_github_file"
DATA = {"repo": "owner/repo", "path": "big.txt", "ref": "main"}


def _host_measure(result: ToolResult) -> int:
    """The host's own measure, written out independently of the code under
    test: json.dumps defaults over to_dict() plus the wrapper's keys."""
    return len(json.dumps({**result.to_dict(), "tool": TOOL, "success": True}))


def _page(text, offset=0, limit=None):
    return paged_text_result(
        text, offset=offset, limit=limit, tool_name=TOOL, subject="big.txt at main", data=DATA
    )


@pytest.fixture(autouse=True)
def default_cap(monkeypatch):
    monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", "8000")


class TestToolResultCap:
    def test_reads_configured_cap(self, monkeypatch):
        monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", "1234")
        assert tool_result_cap() == 1234

    def test_defaults_to_host_default(self, monkeypatch):
        monkeypatch.delenv("KESTREL_MAX_TOOL_RESULT_CHARS")
        assert tool_result_cap() == 8000

    def test_malformed_cap_fails_fast(self, monkeypatch):
        monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", "lots")
        with pytest.raises(ValueError):
            tool_result_cap()


class TestSerializedResultLen:
    def test_matches_host_measure(self):
        result = ToolResult.ok("done", data={"content": 'a"\\\né🦅'})
        assert serialized_result_len(result, TOOL) == _host_measure(result)

    @pytest.mark.parametrize(
        "char, width",
        [("a", 1), ('"', 2), ("\n", 2), ("é", 6), ("日", 6), ("🦅", 12)],
    )
    def test_counts_json_escaped_width(self, char, width):
        empty = ToolResult.ok("done", data={"content": ""})
        one = ToolResult.ok("done", data={"content": char})
        assert serialized_result_len(one, TOOL) - serialized_result_len(empty, TOOL) == width


class TestValidatePage:
    def test_accepts_defaults(self):
        validate_page(0, None)
        validate_page(10, 1)

    def test_rejects_negative_offset(self):
        with pytest.raises(PageRequestError, match="offset must be at least 0"):
            validate_page(-1, None)

    def test_rejects_limit_below_one(self):
        with pytest.raises(PageRequestError, match="limit must be at least 1"):
            validate_page(0, 0)


class TestPagedTextResult:
    def test_small_text_returned_whole(self):
        result = _page("hello")
        assert result.error is None
        assert result.data == {
            **DATA,
            "offset": 0,
            "returned_chars": 5,
            "total_chars": 5,
            "next_offset": None,
            "content": "hello",
        }
        assert "End of file" in result.confirmation

    def test_empty_text(self):
        result = _page("")
        assert result.error is None
        assert result.data["content"] == ""
        assert result.data["returned_chars"] == 0
        assert result.data["total_chars"] == 0
        assert result.data["next_offset"] is None

    def test_limit_within_cap_returned_exactly(self):
        result = _page("abcdefghij", offset=2, limit=3)
        assert result.data["content"] == "cde"
        assert result.data["next_offset"] == 5
        assert "offset=5" in result.confirmation

    def test_limit_past_end_stops_at_end(self):
        result = _page("abcdefghij", offset=8, limit=50)
        assert result.data["content"] == "ij"
        assert result.data["next_offset"] is None

    def test_offset_past_end_is_empty_final_chunk(self):
        result = _page("abc", offset=99)
        assert result.error is None
        assert result.data["content"] == ""
        assert result.data["returned_chars"] == 0
        assert result.data["total_chars"] == 3
        assert result.data["next_offset"] is None
        assert "past the end" in result.confirmation

    def test_chunk_fits_cap_and_is_maximal(self):
        text = "🦅" * 5000
        first = _page(text)
        n = first.data["returned_chars"]
        assert 0 < n < len(text)
        # The room left is less than one more emoji's twelve escaped characters.
        assert 8000 - 12 < _host_measure(first) <= 8000

    def test_limit_shrunk_to_fit_cap(self):
        text = "🦅" * 5000
        result = _page(text, offset=100, limit=4000)
        returned = result.data["returned_chars"]
        assert 0 < returned < 4000
        assert result.data["next_offset"] == 100 + returned
        assert "fewer than limit=4000" in result.confirmation
        assert _host_measure(result) <= 8000

    def test_cap_too_small_for_one_character_fails(self, monkeypatch):
        monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", "50")
        result = _page("abc")
        assert result.error is not None
        assert "the next character" in result.error
        assert "KESTREL_MAX_TOOL_RESULT_CHARS" in result.error

    def test_cap_too_small_for_empty_final_chunk_fails(self, monkeypatch):
        monkeypatch.setenv("KESTREL_MAX_TOOL_RESULT_CHARS", "50")
        result = _page("abc", offset=3)
        assert result.error is not None
        assert "the empty final chunk" in result.error
