"""Watch a trained model play, live, in a pygame window.

    python -m evaluation.visualize --model models/ppo_selfplay.zip \
        --vecnorm models/vecnormalize_selfplay.pkl --opponent evasive --episodes 5

Green jet = the trained agent (ego). Red jet = the scripted opponent. Yellow
dots = bullets. The arena is a torus, so jets that cross an edge reappear on the
opposite side -- that is by design, not a rendering glitch.

Close the window (or press Q / Esc) to stop early.
"""

import argparse

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import SCRIPTED_OPPONENTS
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv


def visualize(model_path, vecnorm_path, config_path, opponent, episodes):
    config = DogfightConfig.from_yaml(config_path)
    opponent_policy = SCRIPTED_OPPONENTS[opponent]

    env = DummyVecEnv(
        [lambda: DogfightEnv(config=config, opponent_policy=opponent_policy, render_mode="human")]
    )
    if vecnorm_path:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, device="cpu")

    quit_requested = False
    for ep in range(episodes):
        if quit_requested:
            break
        obs = env.reset()
        env.envs[0].render()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    quit_requested = True
            if quit_requested:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            env.envs[0].render()
            done = bool(dones[0])

        events = infos[0].get("events", {}) if not quit_requested else {}
        outcome = "win" if events.get("won") else "loss" if events.get("lost") else "timeout"
        print(f"episode {ep + 1}/{episodes} vs {opponent}: {outcome}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_selfplay.zip")
    parser.add_argument("--vecnorm", default="models/vecnormalize_selfplay.pkl")
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--opponent", default="evasive", choices=list(SCRIPTED_OPPONENTS))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    visualize(args.model, args.vecnorm, args.config, args.opponent, args.episodes)


if __name__ == "__main__":
    main()
