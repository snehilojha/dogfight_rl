import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.rule_based import pure_pursuit_policy
from envs.dogfight_env import DogfightEnv


def build_env(config):
    return DogfightEnv(config=config, opponent_policy=pure_pursuit_policy)


def evaluate(model_path, vecnorm_path=None, episodes=10, config_path="training/hyperparams.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

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

    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            episode_reward += float(reward[0])
            episode_length += 1
            done = bool(dones[0])
            final_info = infos[0]

        rewards.append(episode_reward)
        lengths.append(episode_length)

        events = final_info.get("events", {}) if final_info else {}
        if events.get("won", False):
            wins += 1
        elif events.get("lost", False):
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
