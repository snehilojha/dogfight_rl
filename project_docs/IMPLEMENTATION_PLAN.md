# Dogfight RL — Implementation Plan

Goal: take the project from "working PPO-vs-scripted-bot tutorial" to a genuinely good,
legible, reproducible RL project with self-play, honest evaluation, and a proper README.

Phases are ordered so the repo is in a working state after each one. Foundation and
cleanup come first; the risky/interesting work (self-play, reward changes) builds on it.

---

## Phase 0 — Repo hygiene (~½ day)

Everything else builds on a clean repo state.

### 0.1 Untrack artifacts, commit pending deletions

```bash
git rm --cached models/ppo_dogfight.zip models/vecnormalize.pkl
git rm --cached runs/PPO_1/events.out.tfevents.1775580789.nitro.10632.0
git add -A
git commit -m "chore: untrack model/run artifacts, commit pending deletions"
```

- The two already-deleted model files showing in `git status` get committed away here.
- Keep the timestamped checkpoints in `models/` locally (best runs so far) — they are
  gitignored, which is correct.

### 0.2 Delete dead files

| File | Why |
|---|---|
| `test_obs_manual.py`, `test_physics_manual.py` | Superseded by the real pytest suite in `tests/` |
| `agents/ppo_agent.py` | Empty; SB3's `PPO` class *is* the agent — no wrapper needed |
| `notebooks/*.ipynb` (all four, 0 bytes) | Recreated for real in Phase 5 |
| `agents/self_play.py`, `evaluation/benchmark.py`, `evaluation/record.py` | Empty; delete now, re-add with content in Phases 3–4 so no empty file is ever tracked |

### 0.3 Packaging: `pyproject.toml`

```toml
[project]
name = "dogfight-rl"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26,<3",
  "gymnasium>=0.29,<1.2",
  "stable-baselines3>=2.3,<3",
  "torch>=2.2",
  "pygame>=2.5",
  "pyyaml>=6",
  "tensorboard>=2.15",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff"]

[tool.setuptools.packages.find]
include = ["envs*", "agents*", "training*", "evaluation*"]
```

Then:
1. `pip install -e .`
2. Delete every `sys.path.insert(...)` hack (7 files: train.py, callbacks.py,
   evaluate.py, visualize.py, tests/test_env.py, tests/test_physics.py, tests/test_reward.py, tests/test_rule_based.py).
3. Freeze exact working versions: `pip freeze > requirements.txt`
   (pyproject holds loose ranges; requirements.txt holds known-good pins for reproduction).

### 0.4 Un-gitignore documentation

- Remove `context.md`, `Dogfight_AI_Project_Plan.md`, `DOGFIGHT_COLOSSEUM_ROADMAP.md`
  from `.gitignore`.
- Delete `context.md`'s stale "❌ EMPTY" status table (or delete the whole file — the
  Phase 5 README replaces it).

**Done when:** `git status` is clean, `pytest` passes, `python -m training.train --timesteps 1000` runs without the path hacks.

---

## Phase 1 — Single source of truth for config (~½–1 day)

Config defaults are currently duplicated with **conflicting values** across
`dogfight_env.py` (inline dict), `train.py::load_config` (setdefaults), `reward.py`
(`.get` fallbacks), and `hyperparams.yaml` — e.g. `hit_reward` is 1.0 / 4.0 / 7.0
depending on the code path, `min_speed` is 1.0 vs 2.0. Worse, eval envs in
`callbacks.py:75` and `visualize.py:16` are built with **no config at all**, so
evaluation silently runs on different physics than training.

### 1.1 Create `envs/config.py`

```python
import math
from dataclasses import dataclass

@dataclass
class DogfightConfig:
    # arena
    arena_width: float = 800.0
    arena_height: float = 800.0
    # jet
    min_speed: float = 2.0
    max_speed: float = 6.0
    max_turn_rate: float = math.radians(4.0)
    jet_radius: float = 10.0
    max_health: int = 100
    # weapons
    bullet_speed: float = 12.0
    bullet_lifetime: int = 60
    bullet_radius: float = 3.0
    max_cooldown: int = 20
    hit_damage: int = 25
    # episode
    max_steps: int = 2000
    # opponent
    rule_based_throttle: float = 0.55
    fire_cone_angle_deg: float = 15.0
    # reward
    kill_reward: float = 100.0
    death_penalty: float = -100.0
    hit_reward: float = 7.0
    hit_taken_penalty: float = -0.8
    fire_cone_reward: float = 0.2
    closing_shaping_scale: float = 0.5   # see Phase 2.2
    time_penalty: float = -0.04

    @property
    def arena_diag(self) -> float:       # computed, never stored/duplicated
        return math.hypot(self.arena_width, self.arena_height)

    @classmethod
    def from_yaml(cls, path): ...        # ignore unknown keys with a warning

    @classmethod
    def from_dict(cls, d): ...
```

Decisions to encode (canonical values = the YAML ones, since that is what trained the
best model): `min_speed=2.0`, `hit_reward=7.0`, `hit_taken_penalty=-0.8`,
`close_range_distance` semantics replaced in Phase 2.2.

**Delete `out_of_bounds_penalty` entirely** — the arena is toroidal; nothing ever sets
`out_of_bounds`. Dead code.

Add a separate small `TrainConfig` dataclass for PPO hyperparameters (learning_rate,
n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef,
max_grad_norm, total_timesteps, use_vec_normalize, checkpoint_freq, eval_freq,
n_eval_episodes). Split `hyperparams.yaml` into two top-level sections:

```yaml
env:
  arena_width: 800
  ...
training:
  learning_rate: 0.0004
  ...
```

Env physics and optimizer settings are different concerns.

### 1.2 Refactor consumers

- `DogfightEnv.__init__(config: DogfightConfig | dict | None)` — accept dict for
  back-compat, convert via `DogfightConfig.from_dict`. Delete the inline `base_config` dict.
- `train.py::load_config` shrinks to `DogfightConfig.from_yaml(...)` + `TrainConfig.from_yaml(...)`.
- **Fix the eval-env bug**: `DogfightEvalCallback.__init__` takes `config` and passes it
  when building the eval env; `visualize.py` gains a `--config` argument like
  `evaluate.py` already has.

### 1.3 `Jet` / `Bullet` read from config, not literals

`health=100`, `radius=10` are hardcoded in `Jet.__init__` while `max_health` sits in
config; `Bullet.lifetime=60`, `radius=3` likewise (env even overwrites `bullet.lifetime`
after construction — remove that). Pass all of them in, or pass the config object itself.

### 1.4 Tests

New `tests/test_config.py`:
- YAML round-trip.
- Unknown-key warning fires.
- Env built from YAML exposes the YAML values (this test would have caught the eval
  mismatch).

**Done when:** grep for `setdefault(` and hardcoded physics literals returns nothing outside `config.py`; eval callback env and training env are provably identical physics.

---

## Phase 2 — Deduplicate geometry + fix reward shaping (~1 day)

### 2.1 One geometry module

`wrapped_delta` is defined three times (physics.py, reward.py, inlined in
observation.py and `dogfight_env._bullet_hits_jet`); the relative-bearing idiom
(`atan2` + wrap to [-π, π]) is repeated in reward.py, observation.py, rule_based.py.

Consolidate into `envs/physics.py` (natural home, no new file):

```python
def wrap_angle(theta: float) -> float           # exists
def wrapped_delta(delta: float, size: float) -> float   # exists — delete the copies
def toroidal_delta(ax, ay, bx, by, config) -> tuple[float, float]
def relative_bearing(dx, dy, theta) -> float
```

Then:
- `reward.py`: delete its `wrapped_delta` and `toroidal_relative_position`; import from physics.
- `observation.py`: replace the inline wrap block with `toroidal_delta`, the bearing math with `relative_bearing`.
- `dogfight_env.py`: `_bullet_hits_jet` uses `toroidal_delta`; better, delete
  `_bullet_hits_jet` + `_apply_bullet_hits`'s inline collision math and use the
  currently-unused `physics.check_collisions` (one collision implementation, and it
  becomes tested code). The env keeps the damage/health/event bookkeeping, physics owns
  the hit detection.
- `rule_based.py`: import `toroidal_delta` / `relative_bearing` from physics instead of reward.

### 2.2 Fix reward shaping

Problems today:
1. The closing-distance bonus (`reward.py:61`) pays out for *any* distance decrease,
   including the opponent flying toward the agent — the agent gets rewarded for being chased.
2. The unconditional speed reward + fire-cone reward accrue every step, so a timeout
   episode still earns substantial shaped reward (likely why the first model plateaued
   at reward ≈ 66 with long fights).

Fix: replace the closing bonus + speed bonus with **potential-based shaping**, which is
provably policy-invariant (Ng et al. 1999):

```python
phi(s) = -closing_shaping_scale * distance / arena_diag
shaping = gamma * phi(s') - phi(s)
```

Implementation details:
- `compute_reward(events, ego, opp, config)` gains the previous *and* current potential;
  simplest is for the env to compute `phi` before and after stepping and put both in
  `events` (it already threads `prev_distance` through — replace that with `prev_phi`).
- `gamma` must match PPO's gamma — read it from config (one more reason for Phase 1).
- Keep: kill/death terminal rewards, hit/hit-taken, time penalty.
- Keep fire-cone reward but shrink it (or gate it on `distance < some range`) so pure
  circling at long range doesn't farm it. Alternative: drop it entirely once self-play
  works — terminal + hit rewards may suffice.
- Delete: speed reward, out-of-bounds penalty, `close_range_distance`.

### 2.3 Tests

- Extend `tests/test_reward.py`: shaping sums to ~0 over a closed loop (policy
  invariance sanity check); reward for "opponent approaches, agent flees" is ≤ reward
  for "agent approaches".
- `tests/test_physics.py`: cover `toroidal_delta` / `relative_bearing` edge cases
  (wrap across both borders, bearing at ±π).
- Retrain a short run (100k steps) and confirm win rate vs pure pursuit is not worse
  than before the change. Expect faster kills (lower mean episode length).

**Done when:** one definition of each geometry function, `check_collisions` is the only collision code, reward has no dead terms, and a 100k-step sanity run still beats pure pursuit.

---

## Phase 3 — Self-play (~2–3 days, the centerpiece)

The project describes itself as "self-play curriculum" but only ever trains against one
fixed pure-pursuit bot. Once it hits 10/10 there is nothing left to learn. This phase is
what makes the project interesting.

### 3.1 `agents/self_play.py` — opponent pool

```python
class OpponentPool:
    """Snapshots of past policies + scripted baselines, sampled per episode."""
    def __init__(self, pool_dir, config, max_size=20, scripted_prob=0.2): ...
    def add_snapshot(self, model, vecnorm) -> None
    def sample(self) -> OpponentPolicy      # callable(opp_jet, ego_jet, config) -> action
```

Key design points:
- **Snapshot format**: save `model.zip` + the VecNormalize obs stats at snapshot time
  (a frozen copy — the opponent must normalize with *its* stats, not the live training
  stats, or old snapshots see garbage observations).
- **Opponent policy wrapper**: a snapshot opponent needs the observation *from the
  opponent jet's perspective* — `build_obs(opponent_jet, ego_jet, config)` already
  supports this because it's symmetric in its arguments. Wrap:

  ```python
  class SnapshotPolicy:
      def __init__(self, model, obs_rms): ...
      def __call__(self, own_jet, other_jet, config):
          obs = build_obs(own_jet, other_jet, config)
          obs = normalize(obs, self.obs_rms)          # frozen stats
          action, _ = self.model.predict(obs, deterministic=False)
          return action
  ```

  Use `deterministic=False` for opponent diversity. Load snapshot models with
  `device="cpu"` and cache them (LRU, ~5 loaded at once) so sampling is cheap.
- **Pool composition**: always keep `pure_pursuit_policy` (and Phase 4's extra bots) in
  the pool with probability `scripted_prob` — prevents catastrophic forgetting of basics
  and gives a fixed yardstick.
- **Sampling**: uniform over snapshots is fine to start. Optional upgrade:
  prioritized sampling by opponent win rate against the current agent (harder opponents
  sampled more), tracked with an exponential moving average.

### 3.2 Env support: resample opponent per episode

`DogfightEnv.reset()` currently keeps one fixed `opponent_policy`. Add:

```python
def __init__(self, ..., opponent_provider=None):
    # opponent_provider: callable -> policy, called each reset
def reset(...):
    if self.opponent_provider is not None:
        self.opponent_policy = self.opponent_provider()
```

Also randomize spawn positions/headings on reset (currently deterministic quarter-points
facing each other) — with self-play, deterministic spawns invite degenerate opening-move
memorization. Sample positions uniformly with a minimum separation (say ≥ 300 px
toroidal distance) and uniform headings, using `self.np_random` (already seeded via
`super().reset(seed=seed)`).

### 3.3 `training/self_play_train.py` (or a `--self-play` flag in train.py)

Loop structure:

```
1. Warm start: train N_0 steps vs pure pursuit (reuse existing pipeline)  [or load existing checkpoint]
2. add_snapshot(model)
3. repeat for each generation g:
     a. train N steps, opponents sampled from pool each episode
     b. evaluate vs (i) pure pursuit, (ii) latest snapshot, (iii) random pool sample
     c. if win rate vs latest snapshot > threshold (e.g. 55%): add_snapshot(model)
     d. log generation, pool size, win rates to TensorBoard
```

Config additions (`training:` section): `snapshot_win_threshold: 0.55`,
`steps_per_generation: 100_000`, `pool_max_size: 20`, `scripted_opponent_prob: 0.2`.

New callback `SelfPlayCallback` in `training/callbacks.py` handles b–d on an
`eval_freq` cadence, reusing `run_eval_episode`.

### 3.4 Tests

- `OpponentPool.add_snapshot` + `sample` round-trip with a tiny untrained PPO model.
- `SnapshotPolicy` returns a valid action of shape (3,) within action-space bounds.
- Env with `opponent_provider` actually swaps policies between episodes (counter stub).
- Randomized spawns respect min separation and are reproducible under a fixed seed.

### 3.5 Acceptance experiment

Train ~1–2M steps of self-play. Success criteria:
- Later-generation snapshots beat earlier ones (>60% win rate gen N vs gen N-3) —
  demonstrates real curriculum progress; plot this as a generation-vs-generation
  win-rate matrix (great README material).
- Final agent still beats pure pursuit ≥ 9/10.

**Done when:** the acceptance experiment passes and the generation matrix is produced.

---

## Phase 4 — Honest evaluation: benchmark suite + recording (~1 day)

### 4.1 More scripted opponents in `agents/rule_based.py`

Pure pursuit at fixed 0.55 throttle is very exploitable; beating only it proves little.
Add:

- `lead_pursuit_policy` — aims at the target's *projected* position
  (`target_pos + target_v * (distance / bullet_speed)`), fires on the lead angle.
  Strictly harder to dodge than pure pursuit.
- `evasive_policy` — turns perpendicular to the attacker's bearing, randomizes turn
  direction every K steps, full throttle. Tests whether the agent can chase and finish.
- `random_policy` — uniform random actions, as a floor.

Each is the same signature `(own_jet, other_jet, config) -> np.ndarray(3,)`.
Register them in a dict `SCRIPTED_OPPONENTS = {"pure_pursuit": ..., "lead_pursuit": ...,
"evasive": ..., "random": ...}`.

### 4.2 `evaluation/benchmark.py`

CLI: `python -m evaluation.benchmark --model ... --vecnorm ... --config ... --episodes 50 --seeds 0 1 2`

- Runs the model against every opponent in `SCRIPTED_OPPONENTS` (+ optionally a snapshot
  dir), N episodes each, per seed.
- Reports per-opponent: win/loss/timeout rates, mean±std reward, mean episode length.
- Output: pretty table to stdout **and** a `benchmark_results.json` (consumed by the
  Phase 5 notebooks and README table).
- Deduplicate the episode-rollout loop: `evaluate.py`, `callbacks.py`, and benchmark all
  contain the same while-loop — extract one `run_episode(model, env) -> EpisodeResult`
  helper into `evaluation/rollout.py` and use it in all three.

### 4.3 `evaluation/record.py`

CLI: `python -m evaluation.record --model ... --out assets/demo.gif --episodes 1`

- Build env with `render_mode="rgb_array"`, collect frames each step, write GIF via
  `imageio` (add to pyproject `dev` extras or a `media` extra) — or `.mp4` via
  `imageio-ffmpeg` for the README.
- Downsample: keep every 2nd frame, cap ~600 frames, so the GIF stays under a few MB.
- Nice-to-have while here: draw health bars and a step counter in `_draw_scene` so the
  recording is self-explanatory.

### 4.4 Multi-seed training runner

Small script `training/run_seeds.py` (or a Makefile/PowerShell script): loops
`--seed 0..4`, writes models to `models/seed_{i}/`, tensorboard runs to
`runs/seed_{i}/`. Feeds Phase 5's multi-seed analysis. Keep it dumb and sequential.

**Done when:** `benchmark.py` produces a results table across ≥4 opponent types × ≥3 seeds, and `record.py` produces a watchable `assets/demo.gif`.

---

## Phase 5 — Legibility: README, notebooks, writeup (~1 day)

### 5.1 `README.md` (the single highest-leverage artifact)

Structure:

1. **Title + demo GIF** at the very top (`assets/demo.gif`, committed —
   remove `assets/demo.gif` from `.gitignore`).
2. One-paragraph description: what the env is, what the agent learned, toroidal arena,
   PPO + self-play.
3. **Results table** from `benchmark_results.json`: win rate vs each scripted opponent
   (mean±std over seeds) + the self-play generation matrix image.
4. Environment spec: action space, 14-dim observation table, reward terms (one line
   each), physics constants.
5. Quickstart:
   ```bash
   pip install -e ".[dev]"
   pytest
   python -m training.train --config training/hyperparams.yaml
   python -m training.self_play_train
   python -m evaluation.benchmark --model models/...
   python -m evaluation.visualize --model models/...
   ```
6. Project layout (short tree, only real files).
7. Design notes: why potential-based shaping, why frozen-stat snapshots, lessons learned
   (the reward-66 plateau story is genuinely good content).

Fold anything still useful from `context.md` in here; delete `context.md`.

### 5.2 Notebooks (recreate the four deleted ones, now with real content)

- `01_reward_analysis.ipynb` — per-term reward decomposition over rollouts (log each
  term into `info["reward_terms"]` in `compute_reward` — small change, big
  debuggability win); before/after comparison of the Phase 2 shaping fix.
- `02_training_curves.ipynb` — parse tfevents (`tbparse` or `tensorboard.backend`),
  plot win rate / episode length / reward across training for baseline and self-play runs.
- `03_multi_seed_analysis.ipynb` — mean±std curves across the Phase 4.4 seeds; state
  the variance honestly.
- `04_writeup.ipynb` — narrative: approach, self-play curriculum results, generation
  matrix, failure modes observed.

Commit them **executed** (with outputs) so they render on GitHub.

### 5.3 CI (cheap credibility)

`.github/workflows/ci.yml`: on push — `pip install -e ".[dev]"`, `ruff check .`,
`pytest` (headless pygame: set `SDL_VIDEODRIVER=dummy`). Add the badge to the README.

**Done when:** a stranger can go from `git clone` to watching a trained agent fight in under 5 minutes using only the README.

---

## Sequencing & effort summary

| Phase | What | Effort | Depends on |
|---|---|---|---|
| 0 | Repo hygiene, packaging | ½ day | — |
| 1 | Config single source of truth | ½–1 day | 0 |
| 2 | Geometry dedup + reward fix | 1 day | 1 |
| 3 | Self-play | 2–3 days | 1, 2 |
| 4 | Benchmark suite + recording | 1 day | 1 (parallel with 3) |
| 5 | README, notebooks, CI | 1 day | 3, 4 |

Total: roughly 6–8 focused days. Phases 0–2 are pure improvement with no research risk.
Phase 3 is where iteration may be needed (self-play stability); Phase 4's benchmark
suite is deliberately built alongside it so you can measure whether self-play is
actually helping rather than guessing.

## Deliberately out of scope (for now)

- Parallel envs (`SubprocVecEnv`) — nice speedup, but adds Windows-specific pain;
  revisit if self-play training is too slow.
- W&B integration — TensorBoard is sufficient; the project plan mentioned W&B but it
  adds account friction for reproducers.
- Continuous fire action / ammo limits / multi-jet battles — gameplay expansions that
  belong after the self-play core is proven (see DOGFIGHT_COLOSSEUM_ROADMAP.md).
