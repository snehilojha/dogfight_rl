---
name: training-runs
description: How to launch, monitor, and evaluate PPO training runs in this repo — smoke runs, long background runs, TensorBoard, evaluation, and visualization. Use whenever starting a training run or evaluating a checkpoint.
---

# Running training and evaluation

All commands use `./.venv/Scripts/python.exe` from the repo root. Never system Python.

## Smoke run (verify code changes)

```powershell
./.venv/Scripts/python.exe -m training.train --timesteps 2000 --seed 0 --model-out models/smoke --vecnorm-out models/smoke_vecnorm.pkl
```

~1 minute. Confirms env construction, callbacks, save path. Delete the artifacts after.
Use scratch names (`models/smoke*`) — never clobber `models/ppo_dogfight.zip` /
`models/vecnormalize.pkl` (the current best 1M-step model pair) unless the user asks.

## Long runs (100k+): always background

Use the Bash tool with `run_in_background: true` — a 100k run takes ~10–15 min, 1M takes
hours. You are re-invoked when it exits; do not poll or sleep.

```bash
./.venv/Scripts/python.exe -m training.train --timesteps 1000000 --seed 0 \
  --model-out models/ppo_<experiment> --vecnorm-out models/vecnormalize_<experiment>.pkl
```

- Name artifacts after the experiment (`ppo_phase3_gen0`, `vecnormalize_phase3_gen0.pkl`).
- One run at a time (CPU-bound PPO; parallel runs starve each other).
- Progress prints every 5000 steps; eval (10 episodes vs pure pursuit) every 5000 steps
  via `DogfightEvalCallback` — watch `win_rate`, `loss_rate`, `mean_ep_length` in the
  output rather than re-running eval yourself.
- Checkpoints save every 10k steps to the same `--model-out` path (overwriting).

## Interpreting results honestly

- Eval is currently deterministic-spawn + deterministic-policy: all 10 episodes are ONE
  trajectory. Win rate is binary until Phase 3 randomized spawns land.
- Known reference points: 1M steps vs pure pursuit → 10/10 wins. 100k steps → 0/10
  (agent suicide-charges, dies in ~61 steps) under old AND new reward. Neither is a bug.
- `ep_len_mean` converging to ~61 = the agent is charging head-on and losing.
- Report win/loss/timeout rates AND mean episode length, never win rate alone.

## Evaluating a saved checkpoint

```powershell
./.venv/Scripts/python.exe -m evaluation.evaluate --model models/<name>.zip --vecnorm models/<name>_vecnorm.pkl --config training/hyperparams.yaml --episodes 10
```

Model and vecnorm must be the pair saved together (see dogfight-conventions skill).
Visualize with pygame window:

```powershell
./.venv/Scripts/python.exe -m evaluation.visualize --model models/<name>.zip --vecnorm models/<name>_vecnorm.pkl --config training/hyperparams.yaml
```

Visualization opens a window — only run it when the user asks to watch; it is not a
verification step for headless work. For CI/headless contexts set `SDL_VIDEODRIVER=dummy`.

## TensorBoard

Runs log to `runs/`. If the user wants curves:
`./.venv/Scripts/python.exe -m tensorboard.main --logdir runs` (background), then give
them the localhost URL. Don't leave it running after the session's purpose is served.

## Reproducibility rules

- Always pass an explicit `--seed`; default experiments use seed 0, multi-seed studies
  use 0–4.
- When comparing two code versions, hold seed, timesteps, and config identical; the only
  variable is the code (use the worktree method in the reward-invariants skill).
