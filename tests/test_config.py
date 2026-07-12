import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.config import DogfightConfig, TrainConfig, load_configs
from envs.dogfight_env import DogfightEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPERPARAMS = REPO_ROOT / "training" / "hyperparams.yaml"


def test_defaults_are_consistent() -> None:
    config = DogfightConfig()

    assert config.min_speed == 2.0
    assert config.hit_reward == 7.0
    assert math.isclose(config.arena_diag, math.hypot(config.arena_width, config.arena_height))


def test_from_dict_applies_partial_overrides() -> None:
    config = DogfightConfig.from_dict({"max_steps": 5, "hit_reward": 1.0})

    assert config.max_steps == 5
    assert config.hit_reward == 1.0
    assert config.arena_width == 800.0


def test_from_dict_warns_on_unknown_key() -> None:
    with pytest.warns(UserWarning, match="unknown config key 'not_a_real_key'"):
        DogfightConfig.from_dict({"not_a_real_key": 1})


def test_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"env": {"max_speed": 9.0}, "training": {"learning_rate": 0.001}}),
        encoding="utf-8",
    )

    env_config, train_config = load_configs(path)

    assert env_config.max_speed == 9.0
    assert train_config.learning_rate == 0.001


def test_load_configs_supports_flat_legacy_layout(tmp_path) -> None:
    path = tmp_path / "flat.yaml"
    path.write_text(
        yaml.safe_dump({"max_speed": 7.0, "learning_rate": 0.002}),
        encoding="utf-8",
    )

    env_config, train_config = load_configs(path)

    assert env_config.max_speed == 7.0
    assert train_config.learning_rate == 0.002


def test_repo_hyperparams_yaml_loads() -> None:
    env_config, train_config = load_configs(HYPERPARAMS)

    assert isinstance(env_config, DogfightConfig)
    assert isinstance(train_config, TrainConfig)
    assert env_config.min_speed == 2.0
    assert train_config.total_timesteps == 1_000_000


def test_env_built_from_yaml_uses_yaml_values() -> None:
    env_config = DogfightConfig.from_yaml(HYPERPARAMS)
    env = DogfightEnv(config=env_config)
    env.reset()

    assert env.config is env_config
    assert env.ego_jet.v_min == env_config.min_speed
    assert env.ego_jet.health == env_config.max_health
    assert env.ego_jet.radius == env_config.jet_radius


def test_env_accepts_dict_config_for_backcompat() -> None:
    env = DogfightEnv(config={"max_steps": 3})

    assert isinstance(env.config, DogfightConfig)
    assert env.config.max_steps == 3
