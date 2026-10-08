# Prompt Optimization Lab

[![CI](https://github.com/ZhengjunSun/prompt-optimization-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhengjunSun/prompt-optimization-lab/actions/workflows/ci.yml)

A reproducible laboratory for turning prompt engineering into an optimization problem. It searches candidate instructions against a dataset, captures actionable failure feedback, measures output validity, and keeps a quality/cost Pareto frontier.

```text
dataset + metric + seed prompt
           |         |
           v         v
      evaluate -> reflect/mutate -> candidates
           ^                          |
           +------- Pareto select ----+
```

## Why this repository exists

Production prompts should be versioned and tested like code. Manual prompt tweaking hides regressions and rarely accounts for cost. This project makes the loop explicit and can run fully offline with a deterministic model double.

## Included

- Content-addressed prompt candidates for reproducible experiments.
- Exact structured-output scoring, JSON validity, token-shaped cost, and textual failure feedback.
- Seeded evolutionary search with a pluggable model/mutator boundary.
- Multi-objective Pareto selection instead of one opaque aggregate score.
- Tests that demonstrate an initially failing prompt being optimized into a perfect one.

## Run

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Extension path

Use an OpenAI-compatible, Qwen, or local model behind the `Model` callable; replace the deterministic suffix mutator with reflection-driven proposals; store every run as JSONL; then compare GEPA/DSPy, bandit search, and Bayesian optimization under the same evaluation budget.

This is an independent educational implementation inspired by modern prompt optimizers. It does not copy DSPy or TensorZero code.

## License

MIT

