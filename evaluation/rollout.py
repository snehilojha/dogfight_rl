"""Single source of truth for running one evaluation episode.

`evaluate.py`, `training/callbacks.py`, and `evaluation/benchmark.py` all drive
a model through an episode the same way; they share `run_episode` so the rollout
loop and win/loss/timeout classification live in exactly one place.
"""

from dataclasses import dataclass


@dataclass
class EpisodeResult:
    outcome: str  # "win" | "loss" | "timeout"
    total_reward: float
    length: int


def run_episode(model, env, deterministic=True) -> EpisodeResult:
    """Run one episode of ``model`` in a single-env VecEnv and classify the outcome."""
    obs = env.reset()
    done = False
    total_reward = 0.0
    length = 0
    final_info = None

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = env.step(action)
        total_reward += float(reward[0])
        length += 1
        done = bool(dones[0])
        final_info = infos[0]

    events = final_info.get("events", {}) if final_info else {}
    if events.get("won", False):
        outcome = "win"
    elif events.get("lost", False):
        outcome = "loss"
    else:
        outcome = "timeout"

    return EpisodeResult(outcome=outcome, total_reward=total_reward, length=length)
