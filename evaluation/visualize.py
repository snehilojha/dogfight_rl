import argparse
import sys
from pathlib import Path

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.rule_based import pure_pursuit_policy
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv


def build_env(config):
    return DogfightEnv(config=config, opponent_policy=pure_pursuit_policy, render_mode="human")


def visualize(model_path, vecnorm_path=None, config_path="training/hyperparams.yaml"):
    config = DogfightConfig.from_yaml(config_path)
    env = DummyVecEnv([lambda: build_env(config)])

    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)
    obs = env.reset()
    env.envs[0].render()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = env.step(action)
        env.envs[0].render()

        done = done or bool(dones[0])

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_dogfight.zip")
    parser.add_argument("--vecnorm", default="models/vecnormalize.pkl")
    parser.add_argument("--config", default="training/hyperparams.yaml")
    args = parser.parse_args()

    visualize(args.model, args.vecnorm, args.config)


if __name__ == "__main__":
    main()
