import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.rule_based import (
    SCRIPTED_OPPONENTS,
    evasive_policy,
    lead_pursuit_policy,
    pure_pursuit_policy,
)
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv
from evaluation.benchmark import _summarize
from evaluation.rollout import EpisodeResult, run_episode


CONFIG = DogfightConfig()


class MockJet:
    def __init__(self, x, y, theta, v=4.0, w_max=CONFIG.max_turn_rate):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.w_max = w_max


def test_scripted_opponents_registry_complete():
    assert set(SCRIPTED_OPPONENTS) == {"pure_pursuit", "lead_pursuit", "evasive", "random"}


def test_all_opponents_return_valid_actions():
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(400, 300, 1.0)
    for name, policy in SCRIPTED_OPPONENTS.items():
        action = policy(ego, opponent, CONFIG)
        assert action.shape == (3,), name
        assert np.all(np.isfinite(action)), name
        assert -1.0 <= action[0] <= 1.0, name
        assert 0.0 <= action[1] <= 1.0, name
        assert 0.0 <= action[2] <= 1.0, name


def test_lead_pursuit_aims_ahead_of_crossing_target():
    # Opponent directly to the right, moving perpendicular (upward in +y).
    ego = MockJet(100, 100, 0.0)
    opponent = MockJet(300, 100, theta=np.pi / 2, v=6.0)

    pure = pure_pursuit_policy(ego, opponent, CONFIG)
    lead = lead_pursuit_policy(ego, opponent, CONFIG)

    # Pure pursuit aims straight at the current position (turn ~ 0); lead pursuit
    # turns toward where the target is heading.
    assert abs(pure[0]) < 1e-6
    assert lead[0] > 0.0


def test_evasive_reproducible_under_fixed_spawn():
    def rollout():
        jet = MockJet(123.0, 456.0, 0.3)
        attacker = MockJet(400, 400, 0.0)
        return [evasive_policy(jet, attacker, CONFIG, flip_every=2).copy() for _ in range(6)]

    first = rollout()
    second = rollout()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_summarize_computes_mean_and_std():
    per_seed = [
        {"win_rate": 1.0, "loss_rate": 0.0, "timeout_rate": 0.0, "mean_reward": 100.0, "mean_ep_length": 60.0},
        {"win_rate": 0.0, "loss_rate": 1.0, "timeout_rate": 0.0, "mean_reward": -100.0, "mean_ep_length": 40.0},
    ]
    summary = _summarize(per_seed)
    assert summary["win_rate_mean"] == 0.5
    assert summary["win_rate_std"] == 0.5
    assert summary["reward_mean"] == 0.0
    assert summary["ep_length_mean"] == 50.0


class StubModel:
    def predict(self, obs, deterministic=True):
        return np.array([[0.0, 0.5, 0.0]], dtype=np.float32), None


def test_run_episode_returns_episode_result():
    env = DummyVecEnv([lambda: DogfightEnv(config=DogfightConfig(max_steps=5))])
    result = run_episode(StubModel(), env)
    env.close()

    assert isinstance(result, EpisodeResult)
    assert result.outcome in {"win", "loss", "timeout"}
    assert result.length >= 1
