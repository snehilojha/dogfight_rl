---
name: dogfight-conventions
description: Core conventions for working in the dog_fight repo — always read before running Python, training, committing, or touching config. Covers the mandatory venv interpreter, config single-source-of-truth rules, VecNormalize artifact pairing, and commit format.
---

# Dogfight repo conventions

Read this before running any command or editing config/reward code in this repo.

## Python interpreter (non-negotiable)

Every Python invocation MUST use the project venv:

```powershell
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m training.train --timesteps 2000
./.venv/Scripts/python.exe -m evaluation.evaluate --model models/... --vecnorm models/...
```

Never use `python`, `py`, or any system interpreter. The user has explicitly rejected
probing system Pythons. The package is installed editable (`pip install -e .`) into this
venv, so `python -m <pkg>.<module>` works from the repo root with no path hacks — never
add `sys.path.insert` anywhere.

## Config: single source of truth

- `envs/config.py` defines `DogfightConfig` (env/physics/reward) and `TrainConfig` (PPO).
  `training/hyperparams.yaml` has two sections, `env:` and `training:`; `load_configs(path)`
  returns both dataclasses.
- Never hardcode physics or reward constants anywhere else. Never use `.get(key, default)`
  or `setdefault` fallbacks for config — add a dataclass field with a default instead.
- `config.arena_diag` is a computed property; never store it.
- `shaping_gamma` must equal PPO's `gamma`. `train.py` syncs it after loading
  (`env_config.shaping_gamma = train_config.gamma`). Any new training entry point
  (e.g. self-play) MUST replicate that line.
- When adding a config field: add it to the dataclass with a default, add it to
  `hyperparams.yaml`, and rely on `_filter_known`'s unknown-key warning to catch typos.

## Model artifacts

- A model checkpoint is a PAIR: `model.zip` + matching `vecnormalize.pkl`. Loading one
  without the other silently produces garbage observations (VecNormalize normalizes obs
  and reward during training). Every save, load, snapshot, and eval must keep them paired.
- `models/` and `runs/` are gitignored. Never commit artifacts.
- Evaluation must freeze VecNormalize stats (`training=False`, `norm_reward=False`) —
  `DogfightEvalCallback` in `training/callbacks.py` shows the pattern.

## Known evaluation caveat (until Phase 3 lands)

Spawns are deterministic and eval policies are deterministic, so all n eval episodes
are the identical trajectory — win rate is a single binary sample, not a statistic.
Do not draw conclusions from small win-rate differences until randomized spawns exist.
Also: 100k steps vs pure pursuit loses 100% under ANY reward (verified with a baseline
worktree run); only ~1M-step runs beat pure pursuit. Don't misread short runs as
regressions.

## Git

- Commit only when the user asks; the user approves each phase before work starts and
  reviews before commit.
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Conventional-commit style prefixes used so far: `feat:`, `refactor:`, `chore:`, `test:`.

## Module map

| Module | Owns |
|---|---|
| `envs/physics.py` | ALL geometry (`wrap_angle`, `wrapped_delta`, `toroidal_delta`, `relative_bearing`), `Jet`, `Bullet`, and the ONLY collision code (`check_collisions`) |
| `envs/reward.py` | `potential(ego, opp, config)` and `compute_reward(events, ego, opp, config)` |
| `envs/observation.py` | 14-dim symmetric `build_obs(own_jet, other_jet, config)` |
| `envs/dogfight_env.py` | Gymnasium env; threads `prev_phi` through `events` |
| `agents/rule_based.py` | Scripted policies, signature `(own_jet, other_jet, config) -> np.ndarray(3,)` |
| `training/train.py`, `training/callbacks.py` | PPO pipeline, checkpoint/eval callbacks |
| `evaluation/` | evaluate.py, visualize.py (benchmark/record/rollout arrive in Phase 4) |

Never re-implement geometry or collisions locally — import from `envs/physics.py`.
