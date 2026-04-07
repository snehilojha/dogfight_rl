import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.rule_based import pure_pursuit_policy
from envs.dogfight_env import DogfightEnv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = dict(raw)

    if "arena_w" in config:
        config["arena_width"] = config.pop("arena_w")
    if "arena_h" in config:
        config["arena_height"] = config.pop("arena_h")

    config.setdefault("arena_width", 800)
    config.setdefault("arena_height", 800)
    config.setdefault("arena_diag", float(np.sqrt(config["arena_width"] ** 2 + config["arena_height"] ** 2)))
    config.setdefault("min_speed", 1.0)
    config.setdefault("max_speed", 6.0)
    config.setdefault("max_turn_rate", np.deg2rad(4.0))
    config.setdefault("bullet_speed", 12.0)
    config.setdefault("bullet_lifetime", 60)
    config.setdefault("max_cooldown", 20)
    config.setdefault("max_health", 100)
    config.setdefault("hit_damage", 25)
    config.setdefault("max_steps", 2000)

    config.setdefault("total_timesteps", 100_000)
    config.setdefault("learning_rate", 3e-4)
    config.setdefault("n_steps", 2048)
    config.setdefault("batch_size", 256)
    config.setdefault("n_epochs", 10)
    config.setdefault("gamma", 0.99)
    config.setdefault("gae_lambda", 0.95)
    config.setdefault("clip_range", 0.2)
    config.setdefault("ent_coef", 0.01)
    config.setdefault("vf_coef", 0.5)
    config.setdefault("max_grad_norm", 0.5)
    config.setdefault("use_vec_normalize", True)

    return config


def make_env(config):
    def _factory():
        return DogfightEnv(config=config, opponent_policy=pure_pursuit_policy)

    return _factory


def build_vec_env(config):
    env = DummyVecEnv([make_env(config)])

    if config.get("use_vec_normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return env


def build_model(env, config, seed):
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        seed=seed,
        tensorboard_log="runs",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", default="models/ppo_dogfight")
    parser.add_argument("--vecnorm-out", default="models/vecnormalize.pkl")
    args = parser.parse_args()

    config = load_config(args.config)
    total_timesteps = args.timesteps or config["total_timesteps"]

    set_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    env = build_vec_env(config)
    model = build_model(env, config, args.seed)

    print(f"Training for {total_timesteps} timesteps against pure pursuit baseline...")
    model.learn(total_timesteps=total_timesteps)

    model.save(args.model_out)
    if isinstance(env, VecNormalize):
        env.save(args.vecnorm_out)

    print(f"Saved model to {args.model_out}.zip")
    if isinstance(env, VecNormalize):
        print(f"Saved VecNormalize stats to {args.vecnorm_out}")

    env.close()


if __name__ == "__main__":
    main()
