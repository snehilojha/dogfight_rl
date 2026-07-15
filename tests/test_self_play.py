import math

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.rule_based import pure_pursuit_policy
from agents.self_play import OpponentPool, SnapshotPolicy
from envs.config import DogfightConfig
from envs.dogfight_env import DogfightEnv
from envs.physics import Jet, toroidal_delta


def make_jet(x, y, theta, jet_id, config):
    return Jet(
        x, y, theta, jet_id,
        config.min_speed, config.max_speed, config.max_turn_rate,
        config.arena_width, config.arena_height,
    )


def make_vecnorm_env():
    env = DummyVecEnv([lambda: DogfightEnv(config=DogfightConfig(max_steps=50), opponent_policy=pure_pursuit_policy)])
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


@pytest.fixture(scope="module")
def tiny_model():
    # add_snapshot only calls model.save(); the env is irrelevant to snapshotting.
    env = make_vecnorm_env()
    return PPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, device="cpu", seed=0)


# --- OpponentPool / SnapshotPolicy ------------------------------------------


def test_pool_sample_produces_valid_action(tmp_path, tiny_model):
    pool = OpponentPool(tmp_path, DogfightConfig(), scripted_prob=0.0, rng=np.random.default_rng(0))
    pool.add_snapshot(tiny_model, make_vecnorm_env())

    policy = pool.sample()
    config = DogfightConfig()
    action = policy(make_jet(100, 100, 0.0, 1, config), make_jet(400, 400, 0.0, 0, config), config)

    assert action.shape == (3,)
    assert np.all(np.isfinite(action))


def test_pool_reconstructs_from_disk(tmp_path, tiny_model):
    pool = OpponentPool(tmp_path, DogfightConfig(), scripted_prob=0.0)
    pool.add_snapshot(tiny_model, make_vecnorm_env())
    pool.add_snapshot(tiny_model, make_vecnorm_env())

    reloaded = OpponentPool(tmp_path, DogfightConfig(), scripted_prob=0.0)
    assert reloaded.generations == [0, 1]
    assert len(reloaded) == 2


def test_snapshot_uses_frozen_stats(tmp_path, tiny_model):
    env = make_vecnorm_env()
    env.obs_rms.mean[:] = 1.0
    env.obs_rms.var[:] = 4.0

    pool = OpponentPool(tmp_path, DogfightConfig(), scripted_prob=0.0)
    pool.add_snapshot(tiny_model, env)

    # Mutating the live stats after the snapshot must not change the snapshot's
    # normalization — it holds a frozen copy.
    env.obs_rms.mean[:] = 99.0
    env.obs_rms.var[:] = 99.0

    obs = np.zeros(14, dtype=np.float32)
    expected = np.clip((obs - 1.0) / np.sqrt(4.0 + 1e-8), -10.0, 10.0)
    assert np.allclose(pool.latest()._normalize(obs), expected)


def test_pool_evicts_oldest_beyond_max_size(tmp_path, tiny_model):
    pool = OpponentPool(tmp_path, DogfightConfig(), max_size=3, scripted_prob=0.0)
    for _ in range(5):
        pool.add_snapshot(tiny_model, make_vecnorm_env())

    assert len(pool) == 3
    assert pool.generations == [2, 3, 4]
    assert not (tmp_path / "gen_0").exists()


def test_scripted_prob_one_always_returns_scripted(tmp_path, tiny_model):
    def sentinel(own, other, config):
        return np.zeros(3, dtype=np.float32)

    pool = OpponentPool(
        tmp_path, DogfightConfig(), scripted_prob=1.0, scripted_policies=[sentinel], rng=np.random.default_rng(0)
    )
    pool.add_snapshot(tiny_model, make_vecnorm_env())

    for _ in range(10):
        assert pool.sample() is sentinel


def test_empty_pool_samples_scripted(tmp_path):
    pool = OpponentPool(tmp_path, DogfightConfig(), scripted_prob=0.0, rng=np.random.default_rng(0))
    assert pool.latest() is None
    assert pool.sample() is pure_pursuit_policy


def test_snapshot_policy_handles_missing_stats():
    # obs_rms=None => no normalization, obs passed through unchanged.
    class DummyModel:
        def predict(self, obs, deterministic=False):
            return np.asarray(obs[:3], dtype=np.float32), None

    policy = SnapshotPolicy(DummyModel(), obs_rms=None)
    config = DogfightConfig()
    action = policy(make_jet(100, 100, 0.0, 1, config), make_jet(400, 400, 0.0, 0, config), config)
    assert action.shape == (3,)


# --- Env: opponent_provider + randomized spawns -----------------------------


def test_opponent_provider_swaps_each_reset():
    def make_policy():
        def policy(own, other, config):
            return np.array([0.0, 0.5, 0.0], dtype=np.float32)

        return policy

    policies = [make_policy() for _ in range(3)]
    state = {"i": 0}

    def provider():
        policy = policies[state["i"]]
        state["i"] += 1
        return policy

    env = DogfightEnv(config=DogfightConfig(), opponent_provider=provider)
    env.reset()
    first = env.opponent_policy
    env.reset()
    second = env.opponent_policy

    assert state["i"] == 2
    assert first is policies[0]
    assert second is policies[1]


def test_randomized_spawns_respect_min_separation():
    config = DogfightConfig(randomize_spawns=True, min_spawn_separation=300.0)
    env = DogfightEnv(config=config)
    for seed in range(30):
        env.reset(seed=seed)
        dx, dy = toroidal_delta(
            env.ego_jet.x, env.ego_jet.y, env.opponent_jet.x, env.opponent_jet.y,
            config.arena_width, config.arena_height,
        )
        assert math.hypot(dx, dy) >= 300.0 - 1e-6


def test_randomized_spawns_reproducible_under_seed():
    config = DogfightConfig(randomize_spawns=True)

    def pose(env):
        return (env.ego_jet.x, env.ego_jet.y, env.ego_jet.theta,
                env.opponent_jet.x, env.opponent_jet.y, env.opponent_jet.theta)

    e1 = DogfightEnv(config=config)
    e1.reset(seed=42)
    e2 = DogfightEnv(config=config)
    e2.reset(seed=42)
    assert pose(e1) == pose(e2)

    e3 = DogfightEnv(config=config)
    e3.reset(seed=7)
    assert (e1.ego_jet.x, e1.ego_jet.y) != (e3.ego_jet.x, e3.ego_jet.y)


def test_deterministic_spawns_when_disabled():
    config = DogfightConfig(randomize_spawns=False)
    env = DogfightEnv(config=config)
    env.reset()

    assert env.ego_jet.x == config.arena_width * 0.25
    assert env.ego_jet.y == config.arena_height * 0.5
    assert env.opponent_jet.x == config.arena_width * 0.75
    assert math.isclose(env.opponent_jet.theta, math.pi) or math.isclose(env.opponent_jet.theta, -math.pi)
