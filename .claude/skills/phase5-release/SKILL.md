---
name: phase5-release
description: Implementation guide for Phase 5 (release quality) — README with demo GIF and results table, four executed notebooks, and GitHub Actions CI with headless pygame. Use when writing the README, notebooks, or CI for this repo.
---

# Phase 5 — README, notebooks, CI

Full spec in IMPLEMENTATION_PLAN.md §Phase 5. Requires Phases 3–4 artifacts
(`benchmark_results.json`, generation matrix, `assets/demo.gif`). Read
dogfight-conventions first.

## 5.1 README.md (highest-leverage artifact)

Order matters — demo first, results second, how-to third:

1. Title + `assets/demo.gif` at the very top (remove it from `.gitignore`, commit it).
2. One paragraph: 2D toroidal-arena dogfight env, PPO + self-play curriculum, what the
   agent learned.
3. **Results table generated from `benchmark_results.json`** — win rate vs each scripted
   opponent as mean±std over seeds. Never hand-type numbers; write them from the JSON so
   they're reproducible, and state model/commit/seeds under the table. Include the
   generation-matrix image from Phase 3.
4. Environment spec: action space (turn, throttle, fire), 14-dim observation table,
   reward terms one line each (note shaping is potential-based and policy-invariant),
   physics constants from `hyperparams.yaml`.
5. Quickstart (verify EVERY command actually runs before committing):
   ```bash
   pip install -e ".[dev]"
   pytest
   python -m training.train --config training/hyperparams.yaml
   python -m training.self_play_train
   python -m evaluation.benchmark --model models/...
   python -m evaluation.visualize --model models/...
   ```
6. Project layout tree (real files only).
7. Design notes: why potential-based shaping, why frozen-stat snapshots, the reward-66
   plateau story, the 100k-suicide-charge finding (honest lessons > polish).

Fold anything still useful from `context.md` in, then delete `context.md`.

## 5.2 Notebooks (commit EXECUTED, with outputs)

- `01_reward_analysis.ipynb` — per-term decomposition from `info["reward_terms"]`
  (add that logging to `compute_reward` first if absent — terms must sum to the scalar,
  tested); before/after Phase 2 shaping comparison.
- `02_training_curves.ipynb` — parse tfevents (`tbparse`, add to dev extras); win rate /
  episode length / reward for baseline and self-play runs.
- `03_multi_seed_analysis.ipynb` — mean±std curves over `models/seed_{0..4}`; state the
  variance honestly, no cherry-picking the best seed.
- `04_writeup.ipynb` — narrative: approach, self-play results, generation matrix,
  observed failure modes.

Execute headless before committing:
`$env:SDL_VIDEODRIVER='dummy'; ./.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_reward_analysis.ipynb`
(jupyter/nbconvert go in dev extras). A notebook that errors on execute is not done.

## 5.3 CI — `.github/workflows/ci.yml`

- Trigger: push + pull_request. `ubuntu-latest`, Python 3.11 (matches
  `requires-python`), pip cache.
- Steps: `pip install -e ".[dev]"` → `ruff check .` → `pytest`, with
  `SDL_VIDEODRIVER: dummy` in `env:` (pygame headless).
- Torch makes installs slow; if CI time matters, install CPU wheels via
  `--index-url https://download.pytorch.org/whl/cpu` first.
- Before committing, run the same gates locally in the venv:
  `./.venv/Scripts/python.exe -m ruff check .` and `-m pytest` — fix findings, don't
  ignore-list them without user sign-off.
- Add the CI badge to the README once the workflow exists.

## Done when

A stranger goes from `git clone` to watching a trained agent fight in under 5 minutes
using only the README, `ruff` + `pytest` are green in CI, and all four notebooks render
with outputs on GitHub.
