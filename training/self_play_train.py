"""Self-play training: PPO against a growing pool of past snapshots.

The training env samples an opponent from an ``OpponentPool`` each episode. The
pool starts empty, so early episodes are played against scripted baselines (the
warm start); ``SelfPlayCallback`` snapshots the learner as it improves, and the
opponent mix shifts toward those snapshots over time.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from agents.self_play import OpponentPool
from envs.config import load_configs
from training.callbacks import CheckpointWithVecNormCallback, ProgressCallback, SelfPlayCallback
from training.train import build_model, build_vec_env, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-dir", default="models/pool")
    parser.add_argument("--model-out", default="models/ppo_selfplay")
    parser.add_argument("--vecnorm-out", default="models/vecnormalize_selfplay.pkl")
    parser.add_argument("--resume-from", default=None, help="model .zip to warm-start policy weights from")
    parser.add_argument("--resume-vecnorm", default=None, help="VecNormalize .pkl matching --resume-from")
    args = parser.parse_args()

    env_config, train_config = load_configs(args.config)
    # Potential-based shaping is only policy-invariant when its discount matches
    # the learner's; keep this in sync with train.py.
    env_config.shaping_gamma = train_config.gamma
    total_timesteps = args.timesteps or train_config.total_timesteps

    set_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    pool = OpponentPool(
        args.pool_dir,
        env_config,
        max_size=train_config.pool_max_size,
        scripted_prob=train_config.scripted_opponent_prob,
        rng=np.random.default_rng(args.seed),
    )

    env = build_vec_env(env_config, train_config, opponent_provider=pool.sample)
    if args.resume_vecnorm and isinstance(env, VecNormalize):
        env = VecNormalize.load(args.resume_vecnorm, env.venv)
        env.training = True
        env.norm_reward = True

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env, device="cpu")
        print(f"Resumed policy weights from {args.resume_from}")
    else:
        model = build_model(env, train_config, args.seed)

    callbacks = CallbackList(
        [
            ProgressCallback(total_timesteps=total_timesteps, print_freq=5000),
            CheckpointWithVecNormCallback(
                save_freq=train_config.checkpoint_freq,
                model_path=args.model_out,
                vecnorm_path=args.vecnorm_out,
            ),
            SelfPlayCallback(
                pool=pool,
                config=env_config,
                eval_freq=train_config.steps_per_generation,
                n_eval_episodes=train_config.selfplay_eval_episodes,
                snapshot_win_threshold=train_config.snapshot_win_threshold,
                verbose=1,
            ),
        ]
    )

    print(
        f"Self-play training for {total_timesteps} timesteps "
        f"(pool at {args.pool_dir}, snapshot every {train_config.steps_per_generation} steps)..."
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(args.model_out)
    if isinstance(env, VecNormalize):
        env.save(args.vecnorm_out)

    print(f"Saved model to {args.model_out}.zip")
    print(f"Final pool size: {len(pool)} snapshots (generations {pool.generations})")

    env.close()


if __name__ == "__main__":
    main()
