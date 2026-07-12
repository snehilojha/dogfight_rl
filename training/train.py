import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import pure_pursuit_policy
from envs.config import load_configs
from envs.dogfight_env import DogfightEnv
from training.callbacks import CheckpointWithVecNormCallback, DogfightEvalCallback, ProgressCallback


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_config):
    def _factory():
        return DogfightEnv(config=env_config, opponent_policy=pure_pursuit_policy)

    return _factory


def build_vec_env(env_config, train_config):
    env = DummyVecEnv([make_env(env_config)])

    if train_config.use_vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return env


def build_model(env, train_config, seed):
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        n_epochs=train_config.n_epochs,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda,
        clip_range=train_config.clip_range,
        ent_coef=train_config.ent_coef,
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        verbose=1,
        seed=seed,
        tensorboard_log="runs",
        device="cpu",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", default="models/ppo_dogfight")
    parser.add_argument("--vecnorm-out", default="models/vecnormalize.pkl")
    args = parser.parse_args()

    env_config, train_config = load_configs(args.config)
    total_timesteps = args.timesteps or train_config.total_timesteps

    set_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    env = build_vec_env(env_config, train_config)
    model = build_model(env, train_config, args.seed)
    callbacks = CallbackList(
        [
            ProgressCallback(
                total_timesteps=total_timesteps,
                print_freq=5000,
            ),
            CheckpointWithVecNormCallback(
                save_freq=train_config.checkpoint_freq,
                model_path=args.model_out,
                vecnorm_path=args.vecnorm_out,
            ),
            DogfightEvalCallback(
                config=env_config,
                eval_freq=train_config.eval_freq,
                n_eval_episodes=train_config.n_eval_episodes,
            ),
        ]
    )

    print(f"Training for {total_timesteps} timesteps against pure pursuit baseline...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(args.model_out)
    if isinstance(env, VecNormalize):
        env.save(args.vecnorm_out)

    print(f"Saved model to {args.model_out}.zip")
    if isinstance(env, VecNormalize):
        print(f"Saved VecNormalize stats to {args.vecnorm_out}")

    env.close()


if __name__ == "__main__":
    main()
