import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.rule_based import pure_pursuit_policy
from envs.dogfight_env import DogfightEnv


class MockJet:
    def __init__(self, x, y, theta, w_max):
        self.x = x
        self.y = y
        self.theta = theta
        self.w_max = w_max


CONFIG = {
    "arena_width": 800,
    "arena_height": 800,
    "fire_cone_angle_deg": 15.0,
}


def test_pure_pursuit_turns_toward_target() -> None:
    ego = MockJet(100, 100, 0.0, math.radians(4))
    opponent = MockJet(100, 200, 0.0, math.radians(4))

    action = pure_pursuit_policy(ego, opponent, CONFIG)

    assert action[0] > 0
    assert action[1] == 1.0
    assert action[2] == 0.0


def test_pure_pursuit_fires_when_aligned() -> None:
    ego = MockJet(100, 100, 0.0, math.radians(4))
    opponent = MockJet(200, 100, 0.0, math.radians(4))

    action = pure_pursuit_policy(ego, opponent, CONFIG)

    assert np.allclose(action, np.array([0.0, 1.0, 1.0], dtype=np.float32))


def test_env_uses_rule_based_opponent_policy() -> None:
    env = DogfightEnv(opponent_policy=pure_pursuit_policy)
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))

    assert env.opponent_jet.last_action is not None
    assert "events" in info
