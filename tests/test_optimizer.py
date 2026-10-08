import json

from prompt_lab import Candidate, EvolutionOptimizer, Example, Score, pareto_frontier
from prompt_lab.optimizer import evaluate

DATA = [
    Example("alarm: high temperature", {"kind": "alarm"}),
    Example("note: inspection complete", {"kind": "note"}),
]


def fake_model(instruction: str, value: str) -> str:
    if "valid JSON" not in instruction:
        return "kind=alarm"
    return json.dumps({"kind": "alarm" if value.startswith("alarm") else "note"})


def test_evaluate_tracks_quality_format_and_feedback():
    score = evaluate(Candidate("classify"), DATA, fake_model)
    assert score.quality == 0
    assert score.format_rate == 0
    assert len(score.feedback) == 2


def test_optimizer_finds_format_constraint():
    best, history = EvolutionOptimizer(fake_model).optimize("Classify.", DATA, generations=2)
    assert "valid JSON" in best.instruction
    assert max(score.quality for score in history) == 1


def test_candidate_ids_are_content_addressed():
    assert Candidate("x").id == Candidate("x").id
    assert Candidate("x").id != Candidate("y").id


def test_pareto_frontier_keeps_quality_cost_tradeoff():
    scores = [
        Score("a", 1.0, 2.0, 1.0, ()),
        Score("b", 0.8, 1.0, 1.0, ()),
        Score("c", 0.7, 3.0, 1.0, ()),
    ]
    assert [score.candidate_id for score in pareto_frontier(scores)] == ["a", "b"]

