"""Record a trained model playing an episode to an animated GIF.

    python -m evaluation.record --model models/ppo_dogfight.zip \
        --vecnorm models/vecnormalize.pkl --out assets/demo.gif --episodes 1

Runs headless-friendly (set SDL_VIDEODRIVER=dummy). Frames are downsampled to
keep the GIF small enough to commit.
"""

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import SCRIPTED_OPPONENTS
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _normalize_obs(obs, vecnorm):
    if vecnorm is None:
        return obs
    return np.clip(
        (obs - vecnorm.obs_rms.mean) / np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon),
        -vecnorm.clip_obs,
        vecnorm.clip_obs,
    ).astype(np.float32)


def record(model_path, vecnorm_path, out_path, episodes, opponent, config_path, fps, frame_stride, max_frames):
    config = DogfightConfig.from_yaml(config_path)
    opponent_policy = SCRIPTED_OPPONENTS[opponent]

    # A raw (unwrapped) env so we can call render(); normalize obs manually with
    # the frozen VecNormalize stats to match how the model was trained.
    render_env = DogfightEnv(config=config, opponent_policy=opponent_policy, render_mode="rgb_array")
    vecnorm = None
    if vecnorm_path:
        stats_env = DummyVecEnv([lambda: DogfightEnv(config=config, opponent_policy=opponent_policy)])
        vecnorm = VecNormalize.load(vecnorm_path, stats_env)

    model = PPO.load(model_path, device="cpu")

    frames = []
    for _ in range(episodes):
        obs, _ = render_env.reset()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(_normalize_obs(obs, vecnorm), deterministic=True)
            obs, _, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            if step % frame_stride == 0:
                frames.append(render_env.render())
            step += 1
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break

    render_env.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Wrote {out_path} ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_dogfight.zip")
    parser.add_argument("--vecnorm", default="models/vecnormalize.pkl")
    parser.add_argument("--out", default="assets/demo.gif")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--opponent", default="pure_pursuit", choices=list(SCRIPTED_OPPONENTS))
    parser.add_argument("--config", default="training/hyperparams.yaml")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=2, help="keep every Nth frame")
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()

    record(
        args.model, args.vecnorm, args.out, args.episodes, args.opponent,
        args.config, args.fps, args.frame_stride, args.max_frames,
    )


if __name__ == "__main__":
    main()
