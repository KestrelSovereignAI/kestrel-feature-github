"""Classify stalled / blocking GitHub work.

Pure, clock-injectable detection used by the ``scan_stale_work`` tool (and,
later, by a workflow signal source for the kestrel-sovereign#1523 rescue
loop). Given already-fetched GitHub payloads it emits structured
:class:`StaleWorkItem` records. Keeping it free of I/O makes the
classification testable against canned payloads with no network.

Three GitHub-visible classes are detected:
  - ``red_default_branch`` — latest CI run on the default branch failed
    (blocks every merge); highest severity.
  - ``stalled_claim``      — an open issue marked as actively claimed that
    has gone quiet past the staleness threshold.
  - ``stale_pr`` / ``stale_draft`` — an open PR with no update past the
    threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Labels that mark an issue as actively claimed work — the ones whose silence
# means a *stalled claim* rather than an untouched backlog item.
CLAIMED_LABELS = frozenset({"agent-claimed", "in-progress", "in progress"})


@dataclass(frozen=True)
class StaleWorkItem:
    """One piece of stalled or blocking GitHub work."""

    repo: str
    kind: str  # red_default_branch | stalled_claim | stale_pr | stale_draft
    ref: str  # "#123" or a branch name
    title: str
    url: str
    reason: str
    severity: str  # "high" | "medium" | "low"
    age_days: Optional[int] = None


def _parse_ts(value: str) -> datetime:
    """Parse a GitHub ISO-8601 timestamp into an aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _age_days(updated_at: str, now: datetime) -> int:
    return max(0, (now - _parse_ts(updated_at)).days)


def classify_stale_work(
    repo: str,
    *,
    issues: list[dict],
    pull_requests: list[dict],
    latest_default_run: Optional[dict],
    default_branch: str,
    now: datetime,
    stale_days: int = 3,
) -> list[StaleWorkItem]:
    """Classify stalled/blocking work for a single repo from fetched payloads."""
    items: list[StaleWorkItem] = []

    # 1. Red default branch — broken main blocks every merge (highest value).
    if latest_default_run and latest_default_run.get("conclusion") == "failure":
        items.append(
            StaleWorkItem(
                repo=repo,
                kind="red_default_branch",
                ref=default_branch,
                title=str(
                    latest_default_run.get("display_title")
                    or latest_default_run.get("name")
                    or "CI"
                ),
                url=str(latest_default_run.get("html_url") or f"https://github.com/{repo}"),
                reason=f"latest CI run on {default_branch} failed — blocks all merges",
                severity="high",
            )
        )

    # 2. Stalled claims — claimed issues that have gone quiet.
    for issue in issues:
        labels = {lbl.get("name", "").lower() for lbl in issue.get("labels", [])}
        if not (labels & CLAIMED_LABELS):
            continue
        age = _age_days(issue["updated_at"], now)
        if age >= stale_days:
            items.append(
                StaleWorkItem(
                    repo=repo,
                    kind="stalled_claim",
                    ref=f"#{issue['number']}",
                    title=str(issue.get("title") or ""),
                    url=str(issue.get("html_url") or ""),
                    reason=f"claimed but no update in {age}d",
                    severity="high",
                    age_days=age,
                )
            )

    # 3. Stale open PRs — work that opened then stalled (draft or not).
    for pr in pull_requests:
        age = _age_days(pr["updated_at"], now)
        if age < stale_days:
            continue
        is_draft = bool(pr.get("draft"))
        items.append(
            StaleWorkItem(
                repo=repo,
                kind="stale_draft" if is_draft else "stale_pr",
                ref=f"#{pr['number']}",
                title=str(pr.get("title") or ""),
                url=str(pr.get("html_url") or ""),
                reason=f"{'draft ' if is_draft else ''}PR stale {age}d (no update)",
                severity="medium",
                age_days=age,
            )
        )

    return items
