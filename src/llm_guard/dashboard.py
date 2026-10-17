"""View models for the operator dashboard.

The Streamlit app is a thin renderer over these functions so the summaries
can be tested without a browser session.
"""

from collections import Counter
from typing import Any


def request_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the in-memory request counters from an admin snapshot."""

    requests = snapshot.get("requests")
    if not isinstance(requests, dict):
        return []
    rows: list[dict[str, Any]] = []
    for status, count in requests.items():
        rows.append({"status": str(status), "count": int(count)})
    return rows


def cache_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    cache = snapshot.get("cache")
    if not isinstance(cache, dict):
        return []
    return [{"result": str(result), "count": int(count)} for result, count in cache.items()]


def circuit_rows(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    circuits = snapshot.get("circuits")
    if not isinstance(circuits, dict):
        return []
    return [{"model": str(model), "state": str(state)} for model, state in circuits.items()]


def block_counts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count blocked guardrail stages. Gateway summaries are excluded."""

    counter: Counter[str] = Counter()
    for event in events:
        if event.get("decision") != "block":
            continue
        stage = event.get("stage")
        if not isinstance(stage, str) or stage == "gateway":
            continue
        counter[stage] += 1
    return [{"stage": stage, "blocks": count} for stage, count in sorted(counter.items())]


def category_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    categories = report.get("by_category")
    if not isinstance(categories, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in categories:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "category": str(item.get("category", "")),
                "cases": int(item.get("cases", 0)),
                "successes": int(item.get("successes", 0)),
                "rate": float(item.get("rate", 0.0)),
            }
        )
    return rows


def gate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "metric": str(item.get("metric", "")),
                "baseline": float(item.get("baseline", 0.0)),
                "candidate": float(item.get("candidate", 0.0)),
                "delta": float(item.get("delta", 0.0)),
                "ci_low": float(item.get("ci_low", 0.0)),
                "ci_high": float(item.get("ci_high", 0.0)),
                "blocks": bool(item.get("blocks", False)),
            }
        )
    return rows


def headline(snapshot: dict[str, Any], redteam: dict[str, Any] | None, gate: dict[str, Any] | None) -> dict[str, Any]:
    requests = snapshot.get("requests")
    total = 0
    blocked = 0
    if isinstance(requests, dict):
        for status, count in requests.items():
            total += int(count)
            if status in {"guardrail_blocked", "content_filter"}:
                blocked += int(count)
    return {
        "requests": total,
        "blocked": blocked,
        "attack_success_rate": None if redteam is None else redteam.get("attack_success_rate"),
        "false_positive_rate": None if redteam is None else redteam.get("false_positive_rate"),
        "gate": None if gate is None else gate.get("decision"),
    }
