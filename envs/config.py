import math
import warnings
from dataclasses import dataclass, fields

import yaml


def _filter_known(cls, data):
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in known:
            kwargs[key] = value
        else:
            warnings.warn(f"{cls.__name__}: ignoring unknown config key '{key}'", stacklevel=3)
    return kwargs


@dataclass
class DogfightConfig:
    # arena
    arena_width: float = 800.0
    arena_height: float = 800.0

    # jet
    min_speed: float = 2.0
    max_speed: float = 6.0
    max_turn_rate: float = math.radians(4.0)
    jet_radius: float = 10.0
    max_health: int = 100

    # weapons
    bullet_speed: float = 12.0
    bullet_lifetime: int = 60
    bullet_radius: float = 3.0
    max_cooldown: int = 20
    hit_damage: int = 25

    # episode
    max_steps: int = 2000
    # spawns: randomized (uniform positions with a minimum toroidal separation,
    # uniform headings) via the env's seeded RNG. Turn off for fixed scenarios
    # such as demo recording; deterministic spawns place both jets at facing
    # quarter-points.
    randomize_spawns: bool = True
    min_spawn_separation: float = 300.0

    # scripted opponent
    rule_based_throttle: float = 0.55
    fire_cone_angle_deg: float = 15.0

    # reward
    kill_reward: float = 100.0
    death_penalty: float = -100.0
    hit_reward: float = 7.0
    hit_taken_penalty: float = -0.8
    fire_cone_reward: float = 0.2
    closing_shaping_scale: float = 0.5
    # Discount used in the potential-based shaping term; must match the
    # learner's gamma (train.py syncs it from TrainConfig.gamma).
    shaping_gamma: float = 0.99
    time_penalty: float = -0.04

    @property
    def arena_diag(self) -> float:
        return math.hypot(self.arena_width, self.arena_height)

    @classmethod
    def from_dict(cls, data):
        return cls(**_filter_known(cls, data or {}))

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw.get("env", raw))


@dataclass
class TrainConfig:
    total_timesteps: int = 1_000_000
    learning_rate: float = 4e-4
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_vec_normalize: bool = True
    checkpoint_freq: int = 10_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 10

    # self-play (training/self_play_train.py)
    snapshot_win_threshold: float = 0.55
    steps_per_generation: int = 100_000
    pool_max_size: int = 20
    scripted_opponent_prob: float = 0.2
    selfplay_eval_episodes: int = 20

    @classmethod
    def from_dict(cls, data):
        return cls(**_filter_known(cls, data or {}))

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw.get("training", raw))


def load_configs(path):
    """Load (DogfightConfig, TrainConfig) from a YAML file.

    Supports the sectioned layout (top-level `env:` and `training:` keys) and,
    for backwards compatibility, a flat layout where keys from both configs
    live at the top level.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "env" in raw or "training" in raw:
        env_data = raw.get("env", {})
        train_data = raw.get("training", {})
    else:
        env_fields = {f.name for f in fields(DogfightConfig)}
        train_fields = {f.name for f in fields(TrainConfig)}
        env_data = {k: v for k, v in raw.items() if k in env_fields}
        train_data = {k: v for k, v in raw.items() if k in train_fields}
        leftover = set(raw) - env_fields - train_fields
        for key in sorted(leftover):
            warnings.warn(f"load_configs: ignoring unknown config key '{key}'", stacklevel=2)

    return DogfightConfig(**_filter_known(DogfightConfig, env_data)), TrainConfig(
        **_filter_known(TrainConfig, train_data)
    )
