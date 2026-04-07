import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.dogfight_env import DogfightEnv


def test_reset_returns_valid_observation() -> None:
    env = DogfightEnv()
    obs, info = env.reset()

    assert obs.shape == (14,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_gymnasium_tuple() -> None:
    env = DogfightEnv()
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))

    assert obs.shape == (14,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_fire_action_spawns_bullet_and_sets_cooldown() -> None:
    env = DogfightEnv()
    env.reset()

    env.step(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    assert env.ego_jet.gun_cooldown > 0
    assert len(env.bullets) >= 1


def test_env_truncates_at_max_steps() -> None:
    env = DogfightEnv(config={"max_steps": 2})
    env.reset()

    env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is True


def test_rgb_array_render_returns_image() -> None:
    env = DogfightEnv(render_mode="rgb_array")
    env.reset()
    frame = env.render()

    assert frame.shape == (800, 800, 3)
    assert frame.dtype == np.uint8
