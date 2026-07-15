---
name: phase4-benchmark
description: Implementation guide for Phase 4 (honest evaluation) — extra scripted opponents, evaluation/benchmark.py with JSON output, evaluation/rollout.py dedup, GIF recording via evaluation/record.py, and the multi-seed runner. Use when building or modifying the benchmark/recording tooling.
---

# Phase 4 — Benchmark suite + recording

Full spec in IMPLEMENTATION_PLAN.md §Phase 4. Read dogfight-conventions first.
Phase 4 can proceed in parallel with Phase 3 (depends only on Phases 1–2).

## 4.1 Scripted opponents (`agents/rule_based.py`)

All share the signature `(own_jet, other_jet, config) -> np.ndarray(3,)` and must import
geometry from `envs/physics.py` (`toroidal_delta`, `relative_bearing`) — never local math.

- `lead_pursuit_policy` — aim at projected position:
  `lead_point = target + target_velocity * (distance / config.bullet_speed)`, where the
  target's velocity vector is `(v*cos(theta), v*sin(theta))`. Compute the intercept on
  the torus (apply `toroidal_delta` to the lead point). Fire when
  `|relative_bearing|` ≤ fire cone.
- `evasive_policy` — turn perpendicular to the attacker's bearing; re-randomize turn
  direction every K steps (K ~ 20–40) using a per-episode RNG, full throttle, never fires
  or fires rarely. Statefulness trap: module-level state breaks parallel envs — carry
  state on the jet or use a closure/class instantiated per episode.
- `random_policy` — uniform actions as the floor.
- Register: `SCRIPTED_OPPONENTS = {"pure_pursuit": ..., "lead_pursuit": ..., "evasive": ..., "random": ...}`.
- Tests: each returns shape (3,) within action bounds; lead pursuit aims ahead of a
  moving target (bearing to lead point ≠ bearing to target); evasive is reproducible
  under a fixed seed.

## 4.2 `evaluation/rollout.py` — single rollout implementation

Extract the episode while-loop currently duplicated in `evaluation/evaluate.py` and
`training/callbacks.py::run_eval_episode` into:

```python
@dataclass
class EpisodeResult:
    outcome: str          # "win" | "loss" | "timeout"
    total_reward: float
    length: int

def run_episode(model, env, deterministic=True) -> EpisodeResult
```

Then make evaluate.py, callbacks.py, and benchmark.py all consume it. Grep afterwards:
no remaining hand-rolled `while not done` eval loops outside rollout.py.

## 4.3 `evaluation/benchmark.py`

CLI: `./.venv/Scripts/python.exe -m evaluation.benchmark --model M --vecnorm V --config training/hyperparams.yaml --episodes 50 --seeds 0 1 2 [--snapshot-dir models/pool]`

- Model × every `SCRIPTED_OPPONENTS` entry (× optional snapshot dir) × episodes × seeds.
- Per opponent: win/loss/timeout rates, mean±std reward, mean episode length.
- Output BOTH a stdout table and `benchmark_results.json` (machine-readable; consumed by
  Phase 5 notebooks and README). JSON schema: top-level metadata (model path, config,
  episodes, seeds, git commit via `git rev-parse HEAD`, ISO date) + per-opponent stats.
- Eval env must load VecNormalize stats frozen (`training=False`, `norm_reward=False`)
  from the paired pkl.

## 4.4 `evaluation/record.py`

CLI: `./.venv/Scripts/python.exe -m evaluation.record --model M --vecnorm V --out assets/demo.gif --episodes 1`

- Env with `render_mode="rgb_array"`; collect frames per step; write GIF via `imageio`.
- Add `imageio` to a `media` extra in pyproject (`pip install -e ".[media]"` into the venv).
- Downsample: every 2nd frame, cap ~600 frames, target < a few MB.
- Set `SDL_VIDEODRIVER=dummy` so recording works headless.
- While in the renderer: draw health bars + step counter in `_draw_scene` so the GIF is
  self-explanatory.
- `assets/demo.gif` gets committed in Phase 5 (un-gitignore it then, not now).

## 4.5 `training/run_seeds.py`

Dumb sequential loop over `--seed 0..4`; artifacts to `models/seed_{i}/`, logs to
`runs/seed_{i}/`. Hours of runtime — always launch via background Bash (training-runs
skill). No parallelism (Windows + CPU PPO).

## Done when

`benchmark.py` produces a table across ≥4 opponents × ≥3 seeds, `record.py` produces a
watchable `assets/demo.gif`, and rollout logic exists in exactly one place.
