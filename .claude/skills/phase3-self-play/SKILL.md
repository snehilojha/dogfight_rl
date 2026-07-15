---
name: phase3-self-play
description: Implementation guide for Phase 3 (self-play) — OpponentPool, SnapshotPolicy with frozen VecNormalize stats, per-episode opponent resampling, randomized spawns, SelfPlayCallback, and the acceptance experiment. Use when implementing or modifying anything under self-play.
---

# Phase 3 — Self-play

Full spec lives in IMPLEMENTATION_PLAN.md §Phase 3; this skill captures the design
decisions and traps. Read dogfight-conventions and reward-invariants first.

## Components

### `agents/self_play.py`

```python
class OpponentPool:
    def __init__(self, pool_dir, config, max_size=20, scripted_prob=0.2): ...
    def add_snapshot(self, model, vecnorm) -> None
    def sample(self) -> Callable  # (own_jet, other_jet, config) -> np.ndarray(3,)

class SnapshotPolicy:
    def __init__(self, model, obs_rms): ...
    def __call__(self, own_jet, other_jet, config):
        obs = build_obs(own_jet, other_jet, config)   # symmetric — works from either side
        obs = normalize_with_frozen_stats(obs, self.obs_rms)
        action, _ = self.model.predict(obs, deterministic=False)
        return action
```

**Traps that will silently ruin training:**

1. **Frozen stats, not live stats.** Each snapshot must store a deep copy of the
   VecNormalize `obs_rms` at snapshot time and normalize with THAT forever. If snapshots
   share the live training stats, old snapshots receive differently-scaled observations
   than they were trained on and play garbage. Manual normalization formula:
   `clip((obs - obs_rms.mean) / sqrt(obs_rms.var + epsilon), -10, 10)` — match
   VecNormalize's `clip_obs=10.0` and epsilon.
2. **Reward normalization does NOT apply to the opponent** — the opponent only acts, it
   never learns. Only obs normalization matters for snapshots.
3. **`deterministic=False`** for opponent diversity; `device="cpu"` when loading.
4. **LRU cache** ~5 loaded models; the pool dir may hold 20. Loading a PPO zip per reset
   would dominate step time.
5. **Scripted floor**: with probability `scripted_prob`, `sample()` returns
   `pure_pursuit_policy` (later: any of Phase 4's `SCRIPTED_OPPONENTS`) — prevents
   catastrophic forgetting and keeps a fixed yardstick in the mix.
6. Snapshot dir layout: `pool_dir/gen_{k}/model.zip` + `pool_dir/gen_{k}/obs_rms.pkl`.
   Pool state must be reconstructable from the directory alone (resume support).

### Env changes (`envs/dogfight_env.py`)

- Add `opponent_provider: Callable[[], policy] | None` to `__init__`; in `reset()`, if
  set, `self.opponent_policy = self.opponent_provider()`. Keep the existing
  `opponent_policy` arg working (all current tests/training use it).
- **Randomized spawns** in `reset()`: positions uniform over the arena with toroidal
  separation ≥ 300 px (rejection-sample), headings uniform in [-π, π). MUST use
  `self.np_random` (seeded via `super().reset(seed=seed)`) — never `random`/`np.random`
  module state, or reproducibility dies. This change also fixes the deterministic-eval
  problem, so after it lands eval win rates become real statistics.

### `training/self_play_train.py`

Generation loop (warm start → snapshot → train vs pool → gate → snapshot):

- Warm start: N₀ steps vs pure pursuit, or `--resume-from` an existing checkpoint pair.
- Gate: add a snapshot only when win rate vs the latest snapshot > `snapshot_win_threshold`
  (default 0.55). Sync `env_config.shaping_gamma = train_config.gamma` here too.
- New `TrainConfig` fields (+ `training:` YAML keys): `snapshot_win_threshold: 0.55`,
  `steps_per_generation: 100000`, `pool_max_size: 20`, `scripted_opponent_prob: 0.2`.
- `SelfPlayCallback` in `training/callbacks.py` handles eval vs {pure pursuit, latest
  snapshot, random pool sample} on `eval_freq` cadence and logs generation/pool
  size/win rates to TensorBoard. Reuse `run_eval_episode`.

## Required tests (`tests/test_self_play.py`)

- Pool round-trip: `add_snapshot` a tiny untrained PPO (e.g. `n_steps=32`) + fresh
  VecNormalize, `sample()` returns a callable producing shape-(3,) actions within bounds.
- Frozen-stats check: mutate the live VecNormalize after snapshotting; snapshot's
  normalized obs must not change.
- `opponent_provider` swap: counter-stub provider, assert a different policy per reset.
- Randomized spawns: min separation respected over many resets (toroidal distance!);
  identical seeds → identical spawns; different seeds → different spawns.
- Pool eviction at `max_size`; scripted_prob=1.0 always yields the scripted policy.

## Acceptance experiment

Train 1–2M steps of self-play (background, hours — see training-runs skill). Success:

- Generation-vs-generation win-rate matrix: gen N beats gen N−3 with >60% win rate
  (monotone-ish curriculum progress). Save the matrix as JSON + a plot (README asset).
- Final agent still beats pure pursuit ≥ 9/10 (now a real statistic thanks to
  randomized spawns).

Do not declare Phase 3 done on tests alone — the acceptance experiment is the deliverable.
