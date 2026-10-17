"""CLI entry points and small shared helpers."""

from pathlib import Path

import pytest

from llm_guard.cli import main
from llm_guard.sse import chunk_text, completion_chunk, iter_buffered, sse_done
from llm_guard.types import CompletionResult, Message, Usage, estimate_tokens, render_messages, usage_for


def test_token_estimate_and_sse() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert render_messages([Message("user", "hi")]) == "user: hi"
    usage = usage_for((Message("user", "hi"),), "there")
    assert usage.total_tokens >= 2
    with pytest.raises(ValueError):
        chunk_text("abc", 0)
    assert chunk_text("") == []
    result = CompletionResult(
        id="chatcmpl_x",
        created=1,
        model="fake-fast",
        text="Hello",
        usage=Usage(1, 1),
        cost_usd=0.0,
        cache="miss",
        decision="allow",
    )
    lines = list(iter_buffered(result, include_usage=True))
    assert lines[-1] == sse_done()
    assert "Hello" in "".join(lines)
    chunk = completion_chunk(
        completion_id="id",
        created=1,
        model="m",
        delta=None,
        finish_reason="stop",
        usage=Usage(1, 1),
    )
    assert chunk["usage"]["total_tokens"] == 2  # type: ignore[index]


def test_cli_smoke_and_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert main(["redteam", "--smoke", "--output", str(tmp_path / "redteam.json")]) == 0
    assert main(["eval-gate", "--smoke", "--output", str(tmp_path / "eval.json")]) == 0
    report = main(["bench", "--requests", "6", "--concurrency", "3", "--output", str(tmp_path / "bench.json")])
    assert report == 0
    assert (tmp_path / "bench.json").is_file()
    trained = main(
        [
            "train-classifier",
            "--model-output",
            str(tmp_path / "model.json"),
            "--dataset-output",
            str(tmp_path / "data.jsonl"),
        ]
    )
    assert trained == 0
    captured: dict[str, object] = {}

    def run(app: object, host: str, port: int) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", run)
    assert main(["serve", "--host", "127.0.0.1", "--port", "9"]) == 0
    assert captured == {"app": captured["app"], "host": "127.0.0.1", "port": 9}
    with pytest.raises(SystemExit):
        main([])
