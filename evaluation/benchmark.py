"""Benchmark a trained model against every scripted opponent, across seeds.

Emits a human-readable table to stdout and a machine-readable JSON file
(consumed by the notebooks and the README results table).

    python -m evaluation.benchmark --model models/ppo_dogfight.zip \
        --vecnorm models/vecnormalize.pkl --episodes 50 --seeds 0 1 2
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import SCRIPTED_OPPONENTS
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv
from evaluation.rollout import run_episode


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _seed_stats(model, config, vecnorm_path, opponent_policy, episodes, seed):
    env = DummyVecEnv([lambda: DogfightEnv(config=config, opponent_policy=opponent_policy)])
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    env.seed(seed)

    wins = losses = timeouts = 0
    rewards, lengths = [], []
    for _ in range(episodes):
        result = run_episode(model, env)
        rewards.append(result.total_reward)
        lengths.append(result.length)
        if result.outcome == "win":
            wins += 1
        elif result.outcome == "loss":
            losses += 1
        else:
            timeouts += 1
    env.close()

    n = max(1, episodes)
    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "timeout_rate": timeouts / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
    }


def _summarize(per_seed):
    def mean_std(key):
        vals = [s[key] for s in per_seed]
        return float(np.mean(vals)), float(np.std(vals))

    win_mean, win_std = mean_std("win_rate")
    loss_mean, _ = mean_std("loss_rate")
    timeout_mean, _ = mean_std("timeout_rate")
    reward_mean, reward_std = mean_std("mean_reward")
    length_mean, _ = mean_std("mean_ep_length")
    return {
        "win_rate_mean": win_mean,
        "win_rate_std": win_std,
        "loss_rate_mean": loss_mean,
        "timeout_rate_mean": timeout_mean,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "ep_length_mean": length_mean,
    }


def benchmark(model_path, vecnorm_path, config_path, episodes, seeds, out_path):
    config = DogfightConfig.from_yaml(config_path)
    model = PPO.load(model_path, device="cpu")

    results = {}
    for name, opponent_policy in SCRIPTED_OPPONENTS.items():
        per_seed = [
            _seed_stats(model, config, vecnorm_path, opponent_policy, episodes, seed) for seed in seeds
        ]
        results[name] = _summarize(per_seed)

    report = {
        "metadata": {
            "model": model_path,
            "vecnorm": vecnorm_path,
            "config": config_path,
            "episodes_per_seed": episodes,
            "seeds": list(seeds),
            "git_commit": _git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }

    _print_table(report)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {out_path}")
    return report


def _print_table(report):
    seeds = report["metadata"]["seeds"]
    episodes = report["metadata"]["episodes_per_seed"]
    print(f"Benchmark: {episodes} episodes x {len(seeds)} seed(s) per opponent\n")
    header = f"{'opponent':<14}{'win%':>12}{'loss%':>8}{'timeout%':>10}{'reward':>16}{'ep_len':>9}"
    print(header)
    print("-" * len(header))
    for name, s in report["results"].items():
        print(
            f"{name:<14}"
            f"{s['win_rate_mean'] * 100:>6.1f}+/-{s['win_rate_std'] * 100:>3.0f}"
            f"{s['loss_rate_mean'] * 100:>8.1f}"
            f"{s['timeout_rate_mean'] * 100:>10.1f}"
            f"{s['reward_mean']:>9.1f}+/-{s['reward_std']:>4.1f}"
            f"{s['ep_length_mean']:>9.0f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_dogfight.zip")
    parser.add_argument("--vecnorm", default="models/vecnormalize.pkl")
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    benchmark(args.model, args.vecnorm, args.config, args.episodes, args.seeds, args.out)


if __name__ == "__main__":
    main()
