import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import pure_pursuit_policy
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv
from evaluation.rollout import run_episode


def build_env(config):
    return DogfightEnv(config=config, opponent_policy=pure_pursuit_policy)


def evaluate(model_path, vecnorm_path=None, episodes=10, config_path="training/hyperparams.yaml"):
    config = DogfightConfig.from_yaml(config_path)

    env = DummyVecEnv([lambda: build_env(config)])

    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)

    wins = 0
    losses = 0
    timeouts = 0
    rewards = []
    lengths = []

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

    print(f"Episodes: {episodes}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Timeouts: {timeouts}")
    print(f"Win rate: {wins / episodes:.3f}")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Average episode length: {np.mean(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_dogfight.zip")
    parser.add_argument("--vecnorm", default="models/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", default="training/hyperparams.yaml")
    args = parser.parse_args()

    evaluate(args.model, args.vecnorm, args.episodes, args.config)


if __name__ == "__main__":
    main()
