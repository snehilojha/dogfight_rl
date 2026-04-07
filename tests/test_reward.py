import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.reward import compute_reward, toroidal_relative_position


class MockJet:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta


CONFIG = {
    "arena_width": 800,
    "arena_height": 800,
    "kill_reward": 100.0,
    "death_penalty": -100.0,
    "hit_reward": 1.0,
    "hit_taken_penalty": -0.5,
    "fire_cone_reward": 0.3,
    "closing_distance_reward": 0.2,
    "time_penalty": -0.1,
    "out_of_bounds_penalty": -0.2,
    "fire_cone_angle_deg": 15.0,
    "close_range_distance": 150.0,
}


def test_toroidal_relative_position_uses_shortest_path() -> None:
    ego = MockJet(790, 400)
    opponent = MockJet(10, 400)

    dx, dy = toroidal_relative_position(ego, opponent, CONFIG)

    assert dx == 20
    assert dy == 0


def test_compute_reward_terminal_win() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"won": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, 100.2)


def test_compute_reward_terminal_loss() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"lost": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, -99.8)


def test_compute_reward_hit_opponent() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"hit_opponent": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, 1.2)


def test_compute_reward_got_hit() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"got_hit": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, -0.3)


def test_compute_reward_fire_cone_bonus_only_when_aligned() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({}, ego, opponent, CONFIG)

    assert math.isclose(reward, 0.2)


def test_compute_reward_no_fire_cone_bonus_when_outside_angle() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(100, 200, 0.0)

    reward = compute_reward({}, ego, opponent, CONFIG)

    assert math.isclose(reward, -0.1)


def test_compute_reward_closing_distance_bonus() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(260, 100, 0.0)

    reward = compute_reward({"prev_distance": 220.0}, ego, opponent, CONFIG)

    assert math.isclose(reward, 0.4)


def test_compute_reward_no_closing_bonus_when_already_close() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(180, 100, 0.0)

    reward = compute_reward({"prev_distance": 100.0}, ego, opponent, CONFIG)

    assert math.isclose(reward, 0.2)


def test_compute_reward_out_of_bounds_penalty_if_flagged() -> None:
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(200, 100, 0.0)

    reward = compute_reward({"out_of_bounds": True}, ego, opponent, CONFIG)

    assert math.isclose(reward, 0.0, abs_tol=1e-9)
