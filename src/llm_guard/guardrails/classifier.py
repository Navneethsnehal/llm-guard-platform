"""Hashed n-gram logistic regression for prompt-injection detection.

The model is intentionally small and local. Coefficients are JSON on disk,
training is full-batch and seeded, and a holdout split is reported beside
the weights so the file can be regenerated and diffed.
"""

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llm_guard.guardrails.normalize import fold

_TOKEN = re.compile(r"[a-z0-9]+")
FloatArray = NDArray[np.float64]


def featurize(text: str, dim: int) -> FloatArray:
    vector: FloatArray = np.zeros(dim, dtype=np.float64)
    tokens = _TOKEN.findall(fold(text))
    if not tokens:
        return vector
    for token in tokens:
        _add(vector, "w:" + token, 1.0)
        if len(token) >= 3:
            for index in range(len(token) - 2):
                _add(vector, "c:" + token[index : index + 3], 0.35)
    for left, right in itertools.pairwise(tokens):
        _add(vector, "b:" + left + "_" + right, 1.6)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _add(vector: FloatArray, feature: str, weight: float) -> None:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % vector.shape[0]
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += sign * weight


def sigmoid(logits: FloatArray) -> FloatArray:
    clipped = np.clip(logits, -30.0, 30.0)
    result: FloatArray = 1.0 / (1.0 + np.exp(-clipped))
    return result


@dataclass(frozen=True, slots=True)
class HoldoutMetrics:
    precision: float
    recall: float
    f1: float
    fpr: float
    n: int
    threshold: float


@dataclass(frozen=True, slots=True)
class InjectionModel:
    dim: int
    weights: tuple[float, ...]
    bias: float
    threshold: float
    seed: int
    holdout: HoldoutMetrics

    def probability(self, text: str) -> float:
        vector = featurize(text, self.dim)
        weights = np.array(self.weights, dtype=np.float64)
        logit = float(vector @ weights + self.bias)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "dim": self.dim,
            "seed": self.seed,
            "bias": round(self.bias, 6),
            "threshold": round(self.threshold, 6),
            "weights": [round(weight, 6) for weight in self.weights],
            "holdout": {
                "precision": round(self.holdout.precision, 6),
                "recall": round(self.holdout.recall, 6),
                "f1": round(self.holdout.f1, 6),
                "fpr": round(self.holdout.fpr, 6),
                "n": self.holdout.n,
                "threshold": round(self.holdout.threshold, 6),
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> "InjectionModel":
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("classifier file is not an object")
        weights_raw = parsed.get("weights")
        holdout_raw = parsed.get("holdout")
        if not isinstance(weights_raw, list) or not isinstance(holdout_raw, dict):
            raise ValueError("classifier file is missing weights")
        weights = tuple(float(item) for item in weights_raw)
        holdout = HoldoutMetrics(
            precision=float(holdout_raw["precision"]),
            recall=float(holdout_raw["recall"]),
            f1=float(holdout_raw["f1"]),
            fpr=float(holdout_raw["fpr"]),
            n=int(holdout_raw["n"]),
            threshold=float(holdout_raw["threshold"]),
        )
        return cls(
            dim=int(parsed["dim"]),
            weights=weights,
            bias=float(parsed["bias"]),
            threshold=float(parsed["threshold"]),
            seed=int(parsed["seed"]),
            holdout=holdout,
        )

    @classmethod
    def load(cls, path: Path) -> "InjectionModel":
        return cls.from_json(path.read_text(encoding="utf-8"))


def fit_logistic(
    texts: list[str],
    labels: list[int],
    *,
    dim: int,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[FloatArray, float]:
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and aligned")
    matrix = np.stack([featurize(text, dim) for text in texts])
    target = np.array(labels, dtype=np.float64)
    weights: FloatArray = np.zeros(dim, dtype=np.float64)
    bias = 0.0
    count = float(len(texts))
    for _epoch in range(epochs):
        logits = matrix @ weights + bias
        preds = sigmoid(logits)
        error = preds - target
        grad_w = (matrix.T @ error) / count + l2 * weights
        grad_b = float(np.mean(error))
        weights = weights - lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def select_threshold(probabilities: list[float], labels: list[int]) -> float:
    """Highest holdout F1, restricted to FPR <= 5% when that is achievable."""

    ranked: list[tuple[float, float, float]] = []
    for step in range(20, 90):
        threshold = step / 100
        _precision, _recall, f1, fpr = _scores(probabilities, labels, threshold)
        ranked.append((threshold, f1, fpr))
    constrained = [item for item in ranked if item[2] <= 0.05]
    pool = constrained or ranked
    best = max(pool, key=lambda item: (item[1], -item[2], -item[0]))
    return best[0]


def _scores(probabilities: list[float], labels: list[int], threshold: float) -> tuple[float, float, float, float]:
    tp = fp = tn = fn = 0
    for probability, label in zip(probabilities, labels, strict=True):
        predicted = probability >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return precision, recall, f1, fpr


def evaluate(probabilities: list[float], labels: list[int], threshold: float) -> HoldoutMetrics:
    precision, recall, f1, fpr = _scores(probabilities, labels, threshold)
    return HoldoutMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        n=len(labels),
        threshold=threshold,
    )
