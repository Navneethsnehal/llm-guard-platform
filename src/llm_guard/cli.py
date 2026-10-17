"""Command line for red-team, eval gate, training, and benchmarks."""

import argparse
import asyncio
import json
from pathlib import Path

from llm_guard.api.app import create_app
from llm_guard.bench import run_benchmark
from llm_guard.config import Settings, load_settings, repo_root
from llm_guard.evals import GateConfig, compare_scores, evaluate, load_golden
from llm_guard.guardrails.train import build_model, dataset_jsonl
from llm_guard.redteam import run_redteam

DEMO_KEY = "lg_test_regulated_demo_key_0001"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-guard")
    sub = parser.add_subparsers(dest="command", required=True)

    red = sub.add_parser("redteam", help="run the attack suite against the gateway")
    red.add_argument("--smoke", action="store_true")
    red.add_argument("--output", type=Path, default=Path("var/reports/redteam.json"))
    red.add_argument("--config", type=Path, default=None)

    gate = sub.add_parser("eval-gate", help="compare a candidate config to a baseline")
    gate.add_argument("--smoke", action="store_true")
    gate.add_argument("--baseline", type=Path, default=None)
    gate.add_argument("--candidate", type=Path, default=None)
    gate.add_argument("--golden", type=Path, default=None)
    gate.add_argument("--output", type=Path, default=Path("var/reports/eval.json"))

    bench = sub.add_parser("bench", help="measure gateway overhead with the fake provider")
    bench.add_argument("--requests", type=int, default=400)
    bench.add_argument("--concurrency", type=int, default=20)
    bench.add_argument("--output", type=Path, default=Path("var/reports/bench.json"))
    bench.add_argument("--config", type=Path, default=None)

    train = sub.add_parser("train-classifier", help="retrain the injection classifier")
    train.add_argument("--model-output", type=Path, default=repo_root() / "data/models/injection_classifier.json")
    train.add_argument("--dataset-output", type=Path, default=repo_root() / "data/datasets/injection.jsonl")

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "redteam":
        return _redteam(args.smoke, args.output, args.config)
    if args.command == "eval-gate":
        return _eval_gate(args.smoke, args.baseline, args.candidate, args.golden, args.output)
    if args.command == "bench":
        return _bench(args.requests, args.concurrency, args.output, args.config)
    if args.command == "train-classifier":
        return _train(args.model_output, args.dataset_output)
    if args.command == "serve":
        return _serve(args.host, args.port)
    return 2


def _redteam(smoke: bool, output: Path, config: Path | None) -> int:
    app = create_app(load_settings(config))
    report = asyncio.run(run_redteam(app, api_key=DEMO_KEY, smoke=smoke))
    _write(output, report.to_dict())
    print(
        json.dumps(
            {
                "attack_success_rate": report.attack_success_rate,
                "false_positive_rate": report.false_positive_rate,
                "cases": len(report.cases),
            },
            indent=2,
        )
    )
    if report.attack_success_rate > 0 or report.false_positive_rate > 0:
        return 1
    return 0


def _eval_gate(
    smoke: bool,
    baseline: Path | None,
    candidate: Path | None,
    golden: Path | None,
    output: Path,
) -> int:
    golden_path = golden or (repo_root() / "data/evals/golden.jsonl")
    cases = load_golden(golden_path, smoke=smoke)
    base_app = create_app(load_settings(baseline))
    cand_app = create_app(load_settings(candidate))
    base_scores = asyncio.run(evaluate(base_app, cases, api_key=DEMO_KEY))
    cand_scores = asyncio.run(evaluate(cand_app, cases, api_key=DEMO_KEY))
    report = compare_scores(base_scores, cand_scores, GateConfig.default())
    _write(output, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.decision == "pass" else 1


def _bench_settings(config: Path | None) -> Settings:
    """Lift the demo tenant's limiter so the benchmark measures gateway work.

    The published config keeps a 60-request burst. A 400-request run against
    that cap would spend most of its time on HTTP 429, which is not overhead.
    """

    settings = load_settings(config)
    tenants = []
    for tenant in settings.tenants:
        if tenant.id == "regulated-demo":
            tenant = tenant.model_copy(
                update={
                    "requests_per_minute": 100_000,
                    "burst": 10_000,
                    "daily_budget_usd": 10_000,
                }
            )
        tenants.append(tenant)
    return settings.model_copy(update={"tenants": tenants})


def _bench(requests: int, concurrency: int, output: Path, config: Path | None) -> int:
    app = create_app(_bench_settings(config))
    report = asyncio.run(run_benchmark(app, api_key=DEMO_KEY, requests=requests, concurrency=concurrency))
    _write(output, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _train(model_output: Path, dataset_output: Path) -> int:
    model = build_model()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(model.to_json(), encoding="utf-8")
    dataset_output.write_text(dataset_jsonl(), encoding="utf-8")
    print(
        json.dumps(
            {
                "threshold": model.threshold,
                "holdout_f1": model.holdout.f1,
                "holdout_recall": model.holdout.recall,
                "holdout_fpr": model.holdout.fpr,
                "model": str(model_output),
            },
            indent=2,
        )
    )
    return 0


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=host or settings.server.host,
        port=port or settings.server.port,
    )
    return 0


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
