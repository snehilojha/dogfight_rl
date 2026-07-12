# ✈ 2D DOGFIGHT AI

## Full Project Plan & Implementation Guide

Reinforcement Learning · Self-Play · Continuous Control · PPO

## 1. Project Overview

This document is the complete technical blueprint for a 2D Dogfight AI — a portfolio-grade reinforcement learning project where two autonomous jet agents learn to out-manoeuvre and destroy each other in a simulated aerial combat environment.

The project is deliberately designed to stand out on a Data Science / ML resume by combining:

- Continuous action-space PPO — far more technically interesting than discrete gridworld RL
- Self-play multi-agent training — used by OpenAI Five, AlphaStar; shows awareness of real research
- Shaped reward engineering — demonstrates understanding of RL beyond just calling `.learn()`
- Quantitative benchmarking — agent vs. rule-based baseline, proper evaluation metrics
- W&B experiment tracking + paper-style writeup — persistent run history, artifact versioning, and polished dashboard that looks and reads like research, not a tutorial

## 2. Technology Stack

| Domain | Algorithm | Language | Timeline |
| --- | --- | --- | --- |
| Reinforcement Learning | PPO (Stable-Baselines3) | Python 3.11+ | ~4 Weeks |

### 2.1 Core Dependencies

| Library | Version | Purpose |
| --- | --- | --- |
| Python | 3.11+ | Core language — type hints throughout |
| pygame | 2.5+ | 2D rendering engine for simulation & demo visualisation |
| gymnasium | 0.29+ | Standard RL environment API — required for SB3 compatibility |
| stable-baselines3 | 2.3+ | Production-grade PPO implementation — VecEnv, callbacks, checkpoints |
| tensorboard | 2.x | Local low-latency monitoring during active training; zero-config, no network dependency |
| wandb | 0.17+ | Persistent experiment tracking, artifact versioning, multi-seed aggregation, cloud dashboard |
| numpy | 1.26+ | Physics math, observation vectors, reward computation |
| matplotlib | 3.8+ | Trajectory plots, reward curve visualisation in notebooks |
| imageio | 2.34+ | Frame capture and GIF/MP4 export for demo recording |
| pytest | 7.x+ | Unit tests for physics, reward, observation builder |

> **Why keep both?** TensorBoard and W&B serve different roles at different points in the workflow. During active training on a local machine, TensorBoard responds instantly with no network round-trip — useful for catching reward collapse or NaN losses within the first minutes of a run. W&B requires a sync step and a browser tab pointed at wandb.ai. Conversely, W&B is the right tool for everything that outlives a single session: comparing runs across seeds, versioning model artifacts, generating the dashboard screenshots that go in a writeup, and sharing results without committing CSVs to git. Dropping TensorBoard entirely would mean adding network latency to the tight debug loop; dropping W&B entirely would mean losing persistent run history and cross-seed aggregation. `sync_tensorboard=True` in `wandb.init()` means there is zero extra logging code — SB3 writes to TensorBoard as normal and W&B mirrors it automatically.

Why not DQN? DQN requires discrete actions. Discretising jet controls (e.g., 8 turn directions) produces jerky, unrealistic movement and is immediately obvious to any reviewer. PPO's continuous action space allows smooth, fluid manoeuvring that reflects real flight dynamics and is a much stronger technical choice.

## 3. Project Folder Structure

```text
dogfight-ai/
├── envs/
│   ├── dogfight_env.py      # Gymnasium Env — step(), reset(), render()
│   ├── physics.py           # Jet kinematics, bullet travel, collision
│   ├── reward.py            # All reward logic — isolated and testable
│   └── observation.py       # 14-dim state vector construction
│
├── agents/
│   ├── ppo_agent.py         # PPO config, policy kwargs, training entry point
│   ├── rule_based.py        # Deterministic baseline agent for benchmarking
│   └── self_play.py         # Self-play wrapper — opponent pool sampling
│
├── training/
│   ├── train.py             # Main training script — CLI entry point (supports --resume)
│   ├── callbacks.py         # W&B + TensorBoard + CheckpointCallback + EvalCallback
│   └── hyperparams.yaml     # All tunable parameters in one place
│
├── evaluation/
│   ├── evaluate.py          # Win-rate, episode length, reward distribution
│   ├── visualize.py         # Pygame render loop to watch trained agent
│   ├── benchmark.py         # Head-to-head: PPO agent vs. rule-based
│   └── record.py            # GIF/MP4 recording pipeline — renders & exports fight clips
│
├── notebooks/
│   ├── 01_reward_analysis.ipynb
│   ├── 02_training_curves.ipynb
│   ├── 03_multi_seed_analysis.ipynb
│   └── 04_writeup.ipynb
│
├── models/                  # Saved .zip model checkpoints
├── runs/                    # TensorBoard log directory (local only, gitignored)
├── wandb/                   # W&B local run cache (auto-created by wandb, gitignored)
├── assets/                  # Jet & bullet sprites (simple polygons ok)
├── tests/
│   ├── test_physics.py
│   ├── test_reward.py
│   └── test_env.py
├── requirements.txt
└── README.md
```

## 4. Environment Design

### 4.1 Simulation Physics

The physics engine models simplified Newtonian kinematics sufficient for interesting tactical behaviour without requiring a full aerodynamics simulation. Each jet has:

| Property | Value | Notes |
| --- | --- | --- |
| Max speed | 6 px/frame | Normalised in observation |
| Min speed | 1 px/frame | Jets always moving — no hovering |
| Max turn rate | 4°/frame | Proportional to `action[0]` value |
| Bullet speed | 12 px/frame | 2× jet max speed — forces genuine aim lead |
| Bullet lifetime | 60 frames | Limits bullet range, prevents passive camping |
| Gun cooldown | 20 frames | Prevents full-auto spray; forces aim |
| Arena size | 800×800 px | Wrapping toroidal boundary — no edge camping |
| Episode length | Max 2000 steps | Terminates on kill or timeout |

Toroidal (wrap-around) boundaries are an important design choice. Hard walls create degenerate strategies where agents learn to hug corners. Wrapping forces engagement and produces more interesting emergent tactics.

### 4.2 Observation Space (14-dimensional)

The observation vector is constructed relative to the ego agent — positions are normalised to `[-1, 1]`. Angles are always encoded as `(sin, cos)` pairs to avoid the `359° → 0°` discontinuity problem that breaks gradient flow.

```python
# observation.py — build_obs(ego_jet, opponent_jet) -> np.ndarray shape (14,)

# === EGO JET STATE ===
obs[0]  = ego.x / ARENA_W * 2 - 1
obs[1]  = ego.y / ARENA_H * 2 - 1
obs[2]  = sin(ego.heading)
obs[3]  = cos(ego.heading)
obs[4]  = ego.speed / MAX_SPEED

# === OPPONENT RELATIVE STATE ===
obs[5]  = (opp.x - ego.x) / ARENA_W
obs[6]  = (opp.y - ego.y) / ARENA_H
obs[7]  = distance(ego, opp) / ARENA_DIAG
obs[8]  = sin(angle_to_opponent)
obs[9]  = cos(angle_to_opponent)
obs[10] = sin(opp.heading)
obs[11] = cos(opp.heading)

# === TACTICAL STATE ===
obs[12] = ego.gun_cooldown / MAX_COOLDOWN
obs[13] = ego.health / MAX_HEALTH
```

### 4.3 Action Space (Continuous Box)

This 3-dimensional continuous space is why PPO is the correct algorithm choice. The agent must learn fine-grained control — e.g., turning 30% of max rate while holding half throttle — which discrete action spaces cannot express cleanly.

```python
# gymnasium.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
action[0] in [-1.0, 1.0]   # turn rate  : -1 = full left, +1 = full right
action[1] in [0.0, 1.0]    # throttle   : 0 = min speed, +1 = max speed
action[2] in [0.0, 1.0]    # fire intent: > 0.5 = shoot (if cooldown ready)
```

## 5. Reward Engineering

### 5.1 Why Reward Shaping Is Critical

With purely sparse rewards (`+100 kill / -100 die`), the agent wanders randomly for tens of millions of steps before accidentally winning. This makes learning practically impossible on a personal machine. Shaped rewards provide a dense learning signal every single step, guiding the agent toward tactically useful behaviour before it discovers kills.

### 5.2 Full Reward Table

| Event | Value | Justification |
| --- | --- | --- |
| Kill opponent | +100 | Primary objective — sparse, high-magnitude terminal reward |
| Die | -100 | Symmetric negative terminal — agent must learn survival too |
| Bullet hits opponent | +1.0 | Dense shot-reward bridges gap between aim and kill |
| Bullet hits self | -0.5 | Teaches dodging without over-penalising vs. kill objective |
| Opponent in fire cone | +0.3/step | Fire cone = <15° off nose; rewards pursuit and nose-pointing |
| Closing distance | +0.2/step | Only when not already in range; prevents passive orbit |
| Alive (time penalty) | -0.1/step | Forces aggression — agent cannot win by surviving alone |
| Out of bounds | -0.2/step | Discourage boundary-hugging; toroidal arena reduces this need |

The time penalty (`-0.1/step`) is the most important shaped reward. Without it, a trained agent often discovers it can maximise reward by simply not dying — flying evasively forever. The time penalty makes survival alone a losing strategy.

## 6. Self-Play Training

### 6.1 Why Self-Play

Training against a fixed rule-based opponent produces a specialist — an agent optimised to beat exactly that strategy, not a general fighter. Self-play forces the agent to generalise because its opponent improves alongside it.

### 6.2 Opponent Pool Sampling

Implement a frozen opponent pool using periodic checkpointing:

```python
# self_play.py — simplified logic
class SelfPlayOpponentSampler:
    def __init__(self, pool_size=5):
        self.pool = []
        self.pool_size = pool_size

    def update_pool(self, current_model_path):
        self.pool.append(current_model_path)
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)

    def sample_opponent(self):
        # 80% chance of latest opponent, 20% random historical
        if random.random() < 0.8:
            return self.pool[-1]
        return random.choice(self.pool)
```

Update the pool every 50,000 training steps. The 80/20 mix ensures the agent adapts to its current-best version while retaining robustness against earlier strategies — a critical detail that prevents catastrophic forgetting.

## 7. PPO Training Configuration

### 7.1 Recommended Hyperparameters (`hyperparams.yaml`)

```yaml
algorithm: PPO

# Environment
n_envs: 8
total_timesteps: 2_000_000

# PPO core
n_steps: 2048
batch_size: 256
n_epochs: 10
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5

# Policy network
policy: MlpPolicy
net_arch: [256, 256]
activation_fn: tanh

# Normalisation
use_vec_normalize: true
norm_obs: true
norm_reward: true
clip_obs: 10.0
```

VecNormalize is non-negotiable. Without observation normalisation, the 14-dim state vector has features at vastly different scales (positions in hundreds, angles in `[-1, 1]`, cooldown in `[0, 1]`). This destroys gradient flow. Always wrap with `VecNormalize` before calling `.learn()`.

## 8. Training Phases

### Phase 1 — Bootstrap vs. Rule-Based (0 – 500k steps)

Train the PPO agent against a deterministic rule-based opponent. This gives the agent a guaranteed non-random opponent to learn from and establishes basic pursuit-and-shoot behaviour before self-play begins.

- Expected outcome: agent learns to close distance and shoot within first 200k steps
- TensorBoard signal: episode reward increasing from `~-50` to `~+20`

### Phase 2 — Self-Play (500k – 2M steps)

Switch opponent to the self-play pool sampler. The agent now faces improving versions of itself. Expect reward to temporarily dip as the opponent gets harder — this is healthy and expected.

- Update pool every 50k steps
- Watch for collapse: if reward trends negative for >200k steps, reduce learning rate
- Expected emergent tactics: lead turning, defensive spiralling, burst fire, energy management

### Phase 3 — Evaluation & Benchmarking (after 2M steps)

Run formal evaluation: 500 episodes agent vs. rule-based, 500 episodes agent vs. earlier self-play checkpoints. Report win rate, avg episode length, avg total reward.

## 9. Evaluation & Metrics

| Metric | Target | How to measure |
| --- | --- | --- |
| Win rate vs. rule-based | > 85% | 500 eval episodes, `benchmark.py` |
| Win rate vs. self (50k ckpt) | > 65% | 500 episodes vs. `pool[-2]` |
| Avg episode length | < 800 steps | Shorter = more decisive; measures aggression |
| Mean cumulative reward | > 50 per ep | Stable positive signal confirms learning |
| Policy entropy | Gradual decay | Sudden drop = convergence; flat = not learning |
| Value loss | Decreasing | Should plateau by 1M steps |

## 10. Build Sequence (Strict Order)

Do not skip ahead. Each layer depends on the previous. Jumping to PPO training before validating physics and rewards is how projects stall for days on mysterious bugs.

| # | File | What to build | Done when... |
| --- | --- | --- | --- |
| 1 | `envs/physics.py` | Jet movement, bullet travel, collision, wrap boundary | Keyboard-controlled jet flies smoothly, bullets fire correctly |
| 2 | `envs/observation.py` | Build 14-dim state vector from two jet objects | obs values are all in expected ranges, angles use sin/cos |
| 3 | `envs/reward.py` | Implement all 8 reward components in isolation | Unit tests pass; print reward/step and it makes intuitive sense |
| 4 | `envs/dogfight_env.py` | Wrap above 3 in Gymnasium Env; implement `step`/`reset`/`render` with `rgb_array` mode | `check_env()` passes; `render_mode="rgb_array"` returns correct `(H, W, 3)` array |
| 5 | `agents/rule_based.py` | Always-turn-toward + shoot-when-aligned baseline agent | Rule-based agent beats random agent >95% of the time |
| 6 | `training/train.py` | PPO with `SubprocVecEnv`, `VecNormalize`, `--seed`, `--resume`, W&B init; Phase 1 | 500k steps run without crash; W&B dashboard shows rising reward |
| 7 | `training/callbacks.py` | `CheckpointWithNormCallback` saving model + vecnorm + pool; `DogfightWandbCallback` | Checkpoint triplet saved every 50k steps; W&B logs win rate |
| 8 | `agents/self_play.py` | Opponent pool with `save_pool`/`load_pool`; swap to self-play for Phase 2 | 1.5M more steps; agent beats rule-based >85%; pool survives resume |
| 9 | `evaluation/record.py` | GIF/MP4 recording pipeline | `assets/demo.gif` renders cleanly at <5 MB; uploaded to W&B |
| 10 | `evaluation/` | Win rate, episode stats, head-to-head benchmark | Full metrics table populated; visualiser shows trained fight |
| 11 | Seeds 1 & 2 | Re-run training with `--seed 1` and `--seed 2` | 3 W&B runs in group `ppo-selfplay-v1`; results within expected variance |
| 12 | `notebooks/` | Reward analysis, training curves, multi-seed analysis, paper-style writeup | 4 notebooks clean and runnable end-to-end; mean ± std reported |

## 11. Experiment Tracking — W&B + TensorBoard

### 11.1 Why Both?

TensorBoard is fast and local — useful during active training to catch crashes early. W&B persists runs permanently, tracks config diffs between experiments, versions model artifacts, and produces the polished dashboard screenshots that belong in a portfolio writeup.

### 11.2 W&B Setup

```bash
pip install wandb
wandb login   # one-time — stores API key in ~/.netrc, never commit this
```

Set the project name via env var so it never appears hardcoded:

```bash
export WANDB_PROJECT=dogfight-ai
export WANDB_ENTITY=your-username
```

### 11.3 Integration in `callbacks.py`

SB3's `WandbCallback` from `wandb.integration.sb3` handles automatic metric logging. Extend it to log custom metrics:

```python
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

class DogfightWandbCallback(WandbCallback):
    """Extends WandbCallback to log win rate and opponent pool size."""

    def __init__(self, eval_env, eval_freq: int, **kwargs):
        super().__init__(**kwargs)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            win_rate = _compute_win_rate(self.model, self.eval_env, n_episodes=50)
            wandb.log({"custom/win_rate": win_rate}, step=self.num_timesteps)
        return True
```

Always log the full config at run init so every run is reproducible from W&B alone:

```python
wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    config=hyperparams,       # dict loaded from hyperparams.yaml
    sync_tensorboard=True,    # mirrors all SB3 TensorBoard scalars to W&B
    save_code=True,           # snapshots training/train.py in the run
)
```

### 11.4 Launching

```bash
# Terminal 1 — run training (W&B auto-syncs)
python training/train.py --config training/hyperparams.yaml

# Terminal 2 — local TensorBoard (optional, for low-latency monitoring)
tensorboard --logdir runs/

# W&B dashboard: https://wandb.ai/<entity>/dogfight-ai
```

### 11.5 Key Metrics to Monitor

- `rollout/ep_rew_mean` — primary signal. Should increase steadily then plateau
- `rollout/ep_len_mean` — should decrease as agent becomes more decisive
- `train/value_loss` — should decrease; spike = reward instability
- `train/entropy_loss` — should decay slowly; sudden drop = premature convergence
- `train/approx_kl` — should stay < `0.02`; higher means updates too large
- `custom/win_rate` — logged from `DogfightWandbCallback`; most interpretable metric for writeup

### 11.6 Artifact Versioning

Log model checkpoints as W&B artifacts so every saved model is retrievable by run ID:

```python
artifact = wandb.Artifact(name="dogfight-model", type="model")
artifact.add_file("models/best_model.zip")
artifact.add_file("models/vecnorm.pkl")
wandb.log_artifact(artifact)
```

## 12. Resume-From-Checkpoint

### 12.1 Why This Matters

A 2M-step run on a personal machine takes hours. A crash, OS update, or power cut mid-run without resume support means starting over. Resume must be a first-class feature, not an afterthought.

### 12.2 What Must Be Saved Together

A resumable checkpoint is **three files**, not one. Restoring only `model.zip` produces garbage predictions because the observation normalisation statistics are missing:

| File | Saved by | Purpose |
| --- | --- | --- |
| `models/ckpt_{step}.zip` | SB3 `CheckpointCallback` | Policy + value network weights, optimizer state |
| `models/vecnorm_{step}.pkl` | Manual save in callback | Running mean/var for obs and reward normalisation |
| `models/selfplay_pool_{step}.json` | Custom save in `self_play.py` | Opponent pool paths — without this, pool resets to empty |

### 12.3 Saving in `callbacks.py`

```python
class CheckpointWithNormCallback(BaseCallback):
    """Saves model, VecNormalize stats, and self-play pool together at each checkpoint."""

    def __init__(self, save_freq: int, save_path: str, vec_env, self_play_sampler):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = vec_env
        self.self_play_sampler = self_play_sampler

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            self.model.save(f"{self.save_path}/ckpt_{step}")
            self.vec_env.save(f"{self.save_path}/vecnorm_{step}.pkl")
            self.self_play_sampler.save_pool(f"{self.save_path}/selfplay_pool_{step}.json")
        return True
```

### 12.4 Resuming in `train.py`

```bash
python training/train.py --resume models/ckpt_1000000.zip
```

```python
# train.py — resume logic
if args.resume:
    step_str = Path(args.resume).stem.split("_")[-1]
    model = PPO.load(args.resume, env=vec_env)
    vec_env = VecNormalize.load(f"models/vecnorm_{step_str}.pkl", vec_env)
    vec_env.training = True          # re-enable normalisation updates
    vec_env.norm_reward = True
    self_play_sampler.load_pool(f"models/selfplay_pool_{step_str}.json")
    model.set_env(vec_env)
```

### 12.5 Pitfall: `vec_env.training = False` at Eval

During evaluation always set `vec_env.training = False` and `vec_env.norm_reward = False` — the normalisation stats must be frozen during eval or episode rewards are incomparable across runs.

---

## 13. Multi-Seed Evaluation

### 13.1 Why Single-Seed Results Are Scientifically Weak

A single training run can get lucky or unlucky due to random weight initialisation and environment stochasticity. Reporting results from one seed is the most common credibility gap in student ML portfolios. Three seeds with mean ± std is the minimum bar for a result worth presenting.

### 13.2 Running 3 Seeds

```bash
# Run all three seeds sequentially (or in parallel if VRAM/RAM permits)
python training/train.py --config training/hyperparams.yaml --seed 0
python training/train.py --config training/hyperparams.yaml --seed 1
python training/train.py --config training/hyperparams.yaml --seed 2
```

Each run produces its own W&B run and checkpoint directory. Use W&B's **group** feature to aggregate them:

```python
wandb.init(
    ...,
    group="ppo-selfplay-v1",   # all seeds share this group
    job_type="train",
    name=f"seed-{args.seed}",
)
```

### 13.3 Seed Coverage

`--seed` must propagate to **all** random sources:

```python
import random, numpy as np, torch

def set_global_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Also pass `seed=args.seed` to `SubprocVecEnv` via the env factory lambda and to `PPO(seed=args.seed, ...)`.

### 13.4 Aggregation in `notebooks/03_multi_seed_analysis.ipynb`

```python
import wandb
api = wandb.Api()

runs = api.runs(
    path="<entity>/dogfight-ai",
    filters={"group": "ppo-selfplay-v1", "state": "finished"}
)

# Extract win_rate history per run, interpolate to common x-axis, plot mean ± std
```

### 13.5 What to Report

| Metric | Report as |
| --- | --- |
| Win rate vs. rule-based | mean ± std over 3 seeds × 500 eval episodes |
| Mean cumulative reward | mean ± std per seed, final 100k steps |
| Episode length | mean ± std per seed |
| Steps to 70% win rate | median across seeds — measures sample efficiency |

---

## 14. GIF Recording Pipeline

### 14.1 Purpose

A 15–30 second GIF at the top of the README is the single highest-ROI asset in the entire project. Recruiters decide whether to read further within 5 seconds. This section specifies a clean, reproducible recording pipeline.

### 14.2 `evaluation/record.py`

```python
"""
record.py — Render a trained agent fight and export to GIF or MP4.

Usage:
    python evaluation/record.py \\
        --model models/best_model.zip \\
        --vecnorm models/vecnorm_best.pkl \\
        --output assets/demo.gif \\
        --episodes 1 \\
        --fps 30
"""
import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.dogfight_env import DogfightEnv


def record(model_path: str, vecnorm_path: str, output: str,
           n_episodes: int = 1, fps: int = 30) -> None:
    """Load a trained model and render fight frames to a GIF or MP4."""
    env = DummyVecEnv([lambda: DogfightEnv(render_mode="rgb_array")])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    frames = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            frame = env.render()   # returns np.ndarray (H, W, 3) in rgb_array mode
            frames.append(frame)

    suffix = Path(output).suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output, frames, fps=fps, loop=0)
    elif suffix in (".mp4", ".mov"):
        imageio.mimwrite(output, frames, fps=fps, quality=8)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    print(f"Saved {len(frames)} frames → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnorm", required=True)
    parser.add_argument("--output", default="assets/demo.gif")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    record(args.model, args.vecnorm, args.output, args.episodes, args.fps)
```

### 14.3 Requirements for `render_mode="rgb_array"`

`DogfightEnv` must implement `render()` returning an `np.ndarray` of shape `(H, W, 3)` when `render_mode="rgb_array"`. Use `pygame.surfarray.array3d(screen)` and transpose to `(H, W, 3)`:

```python
def render(self) -> np.ndarray | None:
    """Returns an RGB frame for recording, or renders to screen for human mode."""
    if self.render_mode == "rgb_array":
        return np.transpose(
            pygame.surfarray.array3d(self.screen), axes=(1, 0, 2)
        )
    if self.render_mode == "human":
        pygame.display.flip()
    return None
```

### 14.4 GIF Optimisation

Raw pygame frames at 800×800 produce large GIFs. Resize before saving:

```python
import cv2  # or PIL
frames = [cv2.resize(f, (400, 400)) for f in frames]  # 2× size reduction = 4× file size
```

Target: < 5 MB GIF for GitHub README embedding.

### 14.5 Uploading to W&B

```python
wandb.log({"demo/fight": wandb.Video(np.stack(frames), fps=fps, format="gif")})
```

This embeds the GIF directly in the W&B run page — useful for comparing behaviour across checkpoints.

---

## 15. Paper-Style Writeup Structure (`notebooks/04_writeup.ipynb`)

Structure the writeup as a mini research paper — reviewers find this impressive because it demonstrates the ability to think and communicate scientifically.

| Section | Content |
| --- | --- |
| Abstract | 3-sentence summary: problem, approach, key result (e.g., 87% win rate vs. baseline) |
| 1. Introduction | Why dogfighting is an interesting RL problem; continuous control, partial observability, adversarial dynamics |
| 2. Environment | Physics, observation space, action space, termination conditions. Include a rendered screenshot. |
| 3. Algorithm | Why PPO over DQN; continuous control justification; hyperparameter choices |
| 4. Reward Design | Full reward table + ablation: show a training curve with/without time penalty |
| 5. Self-Play | Pool sampling strategy; why 80/20 mix; Elo-style progression plot |
| 6. Results | Win rate table, training curves, episode length, qualitative behaviour description |
| 7. Discussion | What worked, what failed, what you would do with more compute |
| 8. References | PPO paper (Schulman 2017), reward shaping (Ng 1999), OpenAI self-play (2017) |

## 16. Common Pitfalls & How to Avoid Them

| Pitfall | Fix |
| --- | --- |
| Forgetting `VecNormalize` | Always wrap env: `env = VecNormalize(SubprocVecEnv([...]), norm_obs=True, norm_reward=True)` |
| `check_env()` warnings ignored | Treat every warning as a bug. SB3 will silently fail with malformed envs. |
| Angle discontinuity in obs | Never put raw angle in obs. Always use `(sin(angle), cos(angle))` pair. |
| No time penalty in reward | Agent learns to survive indefinitely. Add `-0.1/step` immediately. |
| Self-play collapse | If reward trends negative >200k steps: reduce LR, increase entropy coef, widen pool. |
| Saving model without VecNormalize stats | Always save `vec_env.save('vecnorm.pkl')` alongside `model.zip`. Loading model without it produces garbage. |
| Training on `render=True` | Rendering kills training speed 10–50×. Train headless; render only at evaluation. |
| Resuming without freezing VecNorm stats | Set `vec_env.training = False` during eval/recording. Stats must be frozen or episode rewards are incomparable. |
| W&B API key in source code | Use `wandb login` (stored in `~/.netrc`) or `WANDB_API_KEY` env var. Never hardcode in any file. |
| Single-seed result claims | Always report mean ± std over ≥3 seeds. A single lucky run is not a result. |
| GIF frame order wrong | `pygame.surfarray.array3d` returns `(W, H, 3)` — always transpose to `(H, W, 3)` before passing to imageio. |
| Single-difficulty rule-based baseline | One difficulty tier makes benchmark numbers uninterpretable — does 87% win rate mean the agent is good, or the baseline is trivially bad? Implement at least two tiers: **Pure Pursuit** (always turn toward, shoot when aligned) and **Lead Pursuit** (predicts intercept point, shoots ahead of target). Report win rate against each tier separately. A result like "87% vs. pure pursuit, 71% vs. lead pursuit" is a meaningful claim; "87% vs. rule-based" is not. |

## 17. Installation & Quick Start

```bash
# 1. Clone & create virtual env
git clone https://github.com/yourname/dogfight-ai
cd dogfight-ai
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# requirements.txt
gymnasium==0.29.1
stable-baselines3==2.3.2
pygame==2.5.2
tensorboard==2.16.2
wandb==0.17.0
numpy==1.26.4
matplotlib==3.8.4
imageio==2.34.0
imageio-ffmpeg==0.5.1
pyyaml==6.0.1
pytest==7.4.4

# 3. Configure W&B (one-time)
wandb login
export WANDB_PROJECT=dogfight-ai
export WANDB_ENTITY=your-username

# 4. Run training (seed 0)
python training/train.py --config training/hyperparams.yaml --seed 0

# 5. Resume from checkpoint
python training/train.py --resume models/ckpt_1000000.zip --seed 0

# 6. Watch trained agent
python evaluation/visualize.py --model models/best_model.zip --vecnorm models/vecnorm_best.pkl

# 7. Record GIF
python evaluation/record.py --model models/best_model.zip --vecnorm models/vecnorm_best.pkl --output assets/demo.gif

# 8. TensorBoard (local, optional)
tensorboard --logdir runs/
```

## 18. 4-Week Timeline

| Week | Focus | Deliverables |
| --- | --- | --- |
| Week 1 | Simulation foundation | `physics.py` done + tested, `observation.py`, `reward.py` unit tested, `dogfight_env.py` passing `check_env()`, keyboard-playable demo |
| Week 2 | Training pipeline | `rule_based.py` baseline, `train.py` running with `--seed` + `--resume` flags, W&B integrated, Phase 1 (500k steps) complete, `self_play.py`, Phase 2 (1.5M steps) complete, W&B dashboard populated |
| Week 3 | Multi-seed runs + evaluation | Seeds 1 and 2 complete, `evaluate.py` + `benchmark.py` with metrics, mean ± std results table, `record.py` producing GIF |
| Week 4 | Writeup & polish | 4 notebooks clean and runnable end-to-end, `03_multi_seed_analysis.ipynb` with aggregated curves, README with results table + GIF at top, W&B report linked |

## 19. Why This Stands Out on a Resume

| What you have | What it signals to a reviewer |
| --- | --- |
| Continuous PPO, not discrete DQN | You understand action space design, not just algorithm APIs |
| Self-play multi-agent training | Awareness of real SOTA research (OpenAI Five, AlphaStar) |
| Shaped reward with justification | You understand the exploration-exploitation problem, not just reward hacking |
| Baseline comparison + metrics | Scientific rigour — you validate claims, not just claim it works |
| W&B experiment tracking + paper writeup | You use industry-standard MLOps tooling and can communicate ML work clearly |
| Multi-seed evaluation with mean ± std | You understand statistical validity — rare and impressive in portfolio projects |
| Resume-from-checkpoint | You know production training pipelines; you don't just run notebooks |
| GIF demo in README | Instantly impressive in an interview — they can see it working in 5 seconds |

Build sequence: physics → observation → reward → env → baseline → train → self-play → evaluate → writeup

Record a 30-second GIF of the trained agent fighting and put it at the top of your README. This single asset will get your repo more clicks and stars than any amount of written documentation.
