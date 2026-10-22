"""Operator view of live traffic, blocks, red-team results, and the eval gate."""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from llm_guard.dashboard import (
    block_counts,
    cache_rows,
    category_rows,
    circuit_rows,
    gate_rows,
    headline,
    request_rows,
)

GATEWAY_URL = os.environ.get("LLM_GUARD_URL", "http://127.0.0.1:8080")
ADMIN_KEY = os.environ.get("LLM_GUARD_ADMIN_KEY", "lg_admin_local_dev")
REPORTS = Path(os.environ.get("LLM_GUARD_REPORTS", "var/reports"))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else None


def _snapshot() -> dict[str, Any]:
    response = httpx.get(
        f"{GATEWAY_URL}/admin/snapshot",
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        params={"limit": 200},
        timeout=3.0,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("snapshot was not an object")
    return payload


st.set_page_config(page_title="llm-guard", layout="wide")
st.title("llm-guard platform")
st.caption("Live gateway traffic, guardrail decisions, red-team results, and eval-gate history.")

redteam = _load_json(REPORTS / "redteam.json")
gate = _load_json(REPORTS / "eval.json")

try:
    snapshot = _snapshot()
    error = None
except (httpx.HTTPError, ValueError) as exc:
    snapshot = {"requests": {}, "cache": {}, "circuits": {}, "audit": []}
    error = str(exc)

summary = headline(snapshot, redteam, gate)
columns = st.columns(4)
columns[0].metric("Requests", summary["requests"])
columns[1].metric("Guardrail blocks", summary["blocked"])
attack_rate = summary["attack_success_rate"]
columns[2].metric("Attack success", "n/a" if attack_rate is None else attack_rate)
columns[3].metric("Eval gate", summary["gate"] or "n/a")

if error:
    st.warning(f"Gateway snapshot unavailable ({error}). Start the API or point LLM_GUARD_URL at it.")

left, right = st.columns(2)
with left:
    st.subheader("Requests")
    st.dataframe(request_rows(snapshot), hide_index=True)
    st.subheader("Cache")
    st.dataframe(cache_rows(snapshot), hide_index=True)
with right:
    st.subheader("Circuits")
    st.dataframe(circuit_rows(snapshot), hide_index=True)
    st.subheader("Blocks by stage")
    audit = snapshot.get("audit")
    st.dataframe(block_counts(audit if isinstance(audit, list) else []), hide_index=True)

st.subheader("Red team")
if redteam is None:
    st.info("No report at var/reports/redteam.json. Run `make redteam`.")
else:
    st.dataframe(category_rows(redteam), hide_index=True)

st.subheader("Eval gate")
if gate is None:
    st.info("No report at var/reports/eval.json. Run `make eval`.")
else:
    st.dataframe(gate_rows(gate), hide_index=True)
