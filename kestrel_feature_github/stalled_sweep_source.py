"""The ``fleet_stalled_sweep`` signal source.

This is the source the ``stalled_work_rescue`` workflow's ``detect_stalled``
stage dispatches to (kestrel-feature-workflows / sovereign #1523). It is a
deterministic ACTION source — no LLM cost — whose handler runs the GitHub
stale-work scan over the agent's configured fleet and returns the structured
findings as the signal's action result.

The handler itself is supplied by ``GitHubFeature`` (it needs the live client
and the configured fleet); this module owns the source's schema, redaction,
and throttling contract.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)

SOURCE_NAME = "fleet_stalled_sweep"

ActionHandler = Callable[[dict], Awaitable[Any]]


def _schema(payload: dict) -> dict:
    """Validate/normalize the sweep payload.

    Accepts an optional ``stale_days`` (int >= 1) and an optional ``repos``
    spec (``"fleet"``/``"self"``/comma-separated slugs). Anything else is
    rejected so the handler and audit row see a stable contract.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"fleet_stalled_sweep payload must be a dict, got {type(payload).__name__}"
        )
    extra = set(payload.keys()) - {"stale_days", "repos"}
    if extra:
        raise ValueError(
            f"fleet_stalled_sweep payload has unexpected keys: {sorted(extra)}; "
            "allowed: ['stale_days', 'repos']"
        )

    out: dict[str, Any] = {}
    if "stale_days" in payload:
        try:
            stale_days = int(payload["stale_days"])
        except (TypeError, ValueError):
            raise ValueError(
                f"stale_days must be an integer, got {payload['stale_days']!r}"
            )
        if stale_days < 1:
            raise ValueError(f"stale_days must be >= 1, got {stale_days}")
        out["stale_days"] = stale_days
    if "repos" in payload:
        repos = payload["repos"]
        if not isinstance(repos, str) or not repos.strip():
            raise ValueError("repos must be a non-empty string")
        out["repos"] = repos
    return out


def _redact(payload: dict) -> str:
    """Nothing sensitive — just the sweep knobs."""
    return (
        "fleet_stalled_sweep("
        f"stale_days={payload.get('stale_days', 'default')}, "
        f"repos={payload.get('repos', 'fleet')})"
    )


def build_fleet_stalled_sweep_registration(handler: ActionHandler) -> SourceRegistration:
    """Build the ``fleet_stalled_sweep`` source registration around ``handler``."""
    return SourceRegistration(
        name=SOURCE_NAME,
        schema=_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(per_minute=6, per_hour=60),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(summarize=_redact, store_raw_trusted=True),
        retention_days=30,
    )
