from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    input: str
    expected: dict[str, str]


@dataclass(frozen=True)
class Candidate:
    instruction: str
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", hashlib.sha256(self.instruction.encode()).hexdigest()[:10])


@dataclass(frozen=True)
class Score:
    candidate_id: str
    quality: float
    cost: float
    format_rate: float
    feedback: tuple[str, ...]

    def dominates(self, other: Score) -> bool:
        no_worse = self.quality >= other.quality and self.cost <= other.cost
        strictly_better = self.quality > other.quality or self.cost < other.cost
        return no_worse and strictly_better


Model = Callable[[str, str], str]


def evaluate(candidate: Candidate, dataset: Iterable[Example], model: Model) -> Score:
    examples = list(dataset)
    correct = 0
    formatted = 0
    cost = 0.0
    feedback: list[str] = []
    for example in examples:
        raw = model(candidate.instruction, example.input)
        cost += (len(candidate.instruction) + len(example.input) + len(raw)) / 1000
        try:
            parsed = json.loads(raw)
            formatted += 1
        except json.JSONDecodeError:
            feedback.append(f"{example.input}: output is not JSON")
            continue
        if parsed == example.expected:
            correct += 1
        else:
            feedback.append(f"{example.input}: expected {example.expected}, got {parsed}")
    size = max(len(examples), 1)
    return Score(candidate.id, correct / size, cost, formatted / size, tuple(feedback))


def pareto_frontier(scores: Iterable[Score]) -> list[Score]:
    values = list(scores)
    return sorted(
        [score for score in values if not any(other.dominates(score) for other in values)],
        key=lambda score: (-score.quality, score.cost),
    )


class EvolutionOptimizer:
    """Small deterministic search engine; replace the mutator with an LLM in production."""

    suffixes = (
        " Return only valid JSON.",
        " Check every required field before answering.",
        " Be concise and preserve source terminology.",
        " If evidence is missing, use an empty string.",
    )

    def __init__(self, model: Model, seed: int = 7) -> None:
        self.model = model
        self.random = random.Random(seed)

    def optimize(
        self, initial: str, dataset: Iterable[Example], generations: int = 3, width: int = 4
    ) -> tuple[Candidate, list[Score]]:
        data = list(dataset)
        population = [Candidate(initial)]
        history: list[Score] = []
        for _ in range(generations):
            scores = [evaluate(candidate, data, self.model) for candidate in population]
            history.extend(scores)
            score_by_id = {score.candidate_id: score for score in scores}
            population.sort(
                key=lambda candidate: (
                    -score_by_id[candidate.id].quality,
                    score_by_id[candidate.id].cost,
                )
            )
            parent = population[0]
            choices = self.random.sample(self.suffixes, k=min(width, len(self.suffixes)))
            population = [parent] + [Candidate(parent.instruction + suffix) for suffix in choices]
        final_scores = [evaluate(candidate, data, self.model) for candidate in population]
        history.extend(final_scores)
        best_score = min(final_scores, key=lambda score: (-score.quality, score.cost))
        best = next(candidate for candidate in population if candidate.id == best_score.candidate_id)
        return best, history
