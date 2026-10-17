"""Async overhead benchmark against the in-process gateway."""

import asyncio
import time
from dataclasses import dataclass

import httpx
from httpx import ASGITransport

from llm_guard.evals import percentile


@dataclass(frozen=True, slots=True)
class BenchReport:
    requests: int
    concurrency: int
    success: int
    elapsed_s: float
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "concurrency": self.concurrency,
            "success": self.success,
            "elapsed_s": self.elapsed_s,
            "throughput_rps": self.throughput_rps,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


async def run_benchmark(
    app: object,
    *,
    api_key: str,
    requests: int = 200,
    concurrency: int = 20,
) -> BenchReport:
    if requests < 1 or concurrency < 1:
        raise ValueError("requests and concurrency must be positive")
    payload = {
        "model": "latency",
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "The platinum card APR is 2.9 percent for the first twelve months.",
            },
            {"role": "user", "content": "What is the platinum card APR for the first twelve months?"},
        ],
        "llm_guard": {"cache": False},
    }
    latencies: list[float] = []
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://llm-guard.test") as client:
        started = time.perf_counter()
        cursor = 0
        while cursor < requests:
            batch = min(concurrency, requests - cursor)
            outcomes = list(await asyncio.gather(*[_one(client, api_key, payload) for _index in range(batch)]))
            latencies.extend(outcomes)
            cursor += batch
        elapsed = time.perf_counter() - started
    success = sum(latency >= 0 for latency in latencies)
    good = [latency * 1000 for latency in latencies if latency >= 0]
    if not good:
        raise RuntimeError("benchmark produced no successful requests")
    return BenchReport(
        requests=requests,
        concurrency=concurrency,
        success=success,
        elapsed_s=elapsed,
        throughput_rps=success / elapsed if elapsed else 0.0,
        p50_ms=percentile(good, 50),
        p95_ms=percentile(good, 95),
        p99_ms=percentile(good, 99),
    )


async def _one(client: httpx.AsyncClient, api_key: str, payload: dict[str, object]) -> float:
    started = time.perf_counter()
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    if response.status_code != 200:
        return -1.0
    return time.perf_counter() - started
