"""Ranged reads of a text, each chunk sized to the host's tool-result cap (#11).

The host replaces a tool result whose JSON encoding is longer than its
tool-result cap with a head-and-tail preview, so a tool that returns a whole
large file loses the file's middle and nothing can bring it back. A paged read
returns one chunk per call instead: the longest that fits under the cap
together with its envelope, measured the way the host measures it.

Each chunk reports where it starts (``offset``), how many characters it holds
(``returned_chars``), the length of the whole text (``total_chars``) and where
the next chunk starts (``next_offset``, ``None`` at the end). Offsets count the
characters (code points) of the text, so the chunks of one text, joined in
order, are that text exactly. An ``offset`` at or past the end returns an empty
final chunk rather than an error.
"""

import json
import os
from typing import Any, Mapping, Optional

from kestrel_sdk.tools.result import ToolResult

# The host's own default for KESTREL_MAX_TOOL_RESULT_CHARS.
_DEFAULT_TOOL_RESULT_CAP = 8000


class PageRequestError(ValueError):
    """A paged read's ``offset`` or ``limit`` is outside its valid range."""


def tool_result_cap() -> int:
    """The host's per-tool-result cap, ``KESTREL_MAX_TOOL_RESULT_CHARS``.

    Read from the variable the host reads, so the two cannot drift: a result
    whose JSON encoding is longer than the cap reaches the model only as a
    preview.
    """
    return int(
        os.getenv("KESTREL_MAX_TOOL_RESULT_CHARS", str(_DEFAULT_TOOL_RESULT_CAP))
    )


def serialized_result_len(result: ToolResult, tool_name: str) -> int:
    """Length of ``result`` exactly as the host measures it against the cap.

    The host measures ``json.dumps`` (default ``ensure_ascii``) of the envelope
    the tool wrapper builds: ``to_dict()`` plus ``tool`` and ``success``.
    Escaping makes a non-ASCII character six characters long and one outside
    the BMP twelve (a ``\\uXXXX`` surrogate pair), so a raw character count
    understates the size several-fold.
    """
    return len(json.dumps({**result.to_dict(), "tool": tool_name, "success": True}))


def validate_page(offset: int, limit: Optional[int]) -> None:
    """Reject an ``offset`` below 0 or a ``limit`` below 1."""
    if offset < 0:
        raise PageRequestError(f"offset must be at least 0, got {offset}")
    if limit is not None and limit < 1:
        raise PageRequestError(f"limit must be at least 1, got {limit}")


def paged_text_result(
    text: str,
    *,
    offset: int,
    limit: Optional[int],
    tool_name: str,
    subject: str,
    data: Mapping[str, Any],
) -> ToolResult:
    """One chunk of ``text``, as a result that fits the tool-result cap.

    Args:
        text: The whole text being read.
        offset: Character offset the chunk starts at (already validated).
        limit: Most characters to return, or ``None`` for as many as fit. A
            limit is shrunk further when its chunk would not fit the cap.
        tool_name: The calling tool, named in the "call again" instruction and
            counted in the result's measured size.
        subject: What the text is (``"README.md at main"``), for messages.
        data: Fields identifying the text that every result carries ahead of
            the position fields and the chunk.

    While characters remain, a chunk holds at least one, so following
    ``next_offset`` always reaches the end. A cap too small for even that
    fails the read and names the cap, since the host would replace any larger
    result with its preview.
    """
    total = len(text)
    start = min(offset, total)
    requested_end = total if limit is None else min(start + limit, total)

    def build(end: int) -> ToolResult:
        chunk = text[start:end]
        has_more = end < total
        if offset > total:
            confirmation = (
                f"offset {offset} is past the end of {subject} "
                f"({total} characters). End of file."
            )
        else:
            confirmation = f"Read {subject}: characters {start}–{end} of {total}."
            if limit is not None and end < requested_end:
                confirmation += (
                    f" That is fewer than limit={limit}, to fit the tool-result cap."
                )
            if has_more:
                confirmation += (
                    f" More remains; call {tool_name} again with offset={end} "
                    "for the next chunk."
                )
            else:
                confirmation += " End of file."
        return ToolResult.ok(
            confirmation=confirmation,
            data={
                **data,
                "offset": start,
                "returned_chars": len(chunk),
                "total_chars": total,
                "next_offset": end if has_more else None,
                "content": chunk,
            },
        )

    cap = tool_result_cap()
    # Serialized, every character takes at least one character, so no chunk
    # longer than the cap can fit: that bounds both the whole-window attempt
    # and the search, so a large file is never serialized whole.
    if requested_end - start <= cap:
        whole = build(requested_end)
        if serialized_result_len(whole, tool_name) <= cap:
            return whole

    # Binary-search the largest end short of the requested window whose
    # serialized result fits. Every candidate has more remaining, so its
    # envelope only grows with ``end`` and the search is monotone.
    best: Optional[ToolResult] = None
    lo, hi = start + 1, min(requested_end - 1, start + cap)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = build(mid)
        if serialized_result_len(candidate, tool_name) <= cap:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        what = "the next character" if start < requested_end else "the empty final chunk"
        return ToolResult.failed(
            error=(
                f"{tool_name} cannot fit {what} of {subject} under the "
                f"{cap:,}-character tool-result cap (KESTREL_MAX_TOOL_RESULT_CHARS)."
            ),
            data={**data, "offset": start, "total_chars": total},
        )
    return best
