import math

import numpy as np

from envs.physics import relative_bearing, toroidal_delta, wrap_angle


def pure_pursuit_policy(ego_jet, opponent_jet, config):
    """Aim straight at the opponent's current position and fire in the cone."""
    dx, dy = toroidal_delta(
        ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, config.arena_width, config.arena_height
    )
    diff = relative_bearing(dx, dy, ego_jet.theta)

    turn = max(-1.0, min(1.0, diff / ego_jet.w_max))
    fire = 1.0 if abs(diff) <= math.radians(config.fire_cone_angle_deg) else 0.0

    return np.array([turn, config.rule_based_throttle, fire], dtype=np.float32)


def lead_pursuit_policy(ego_jet, opponent_jet, config):
    """Aim at the opponent's projected position (lead the target).

    Estimates where the opponent will be when a bullet arrives and aims there,
    which is strictly harder to dodge than pure pursuit against a moving target.
    """
    dx0, dy0 = toroidal_delta(
        ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, config.arena_width, config.arena_height
    )
    time_to_target = math.hypot(dx0, dy0) / config.bullet_speed

    lead_x = (opponent_jet.x + opponent_jet.v * math.cos(opponent_jet.theta) * time_to_target) % config.arena_width
    lead_y = (opponent_jet.y + opponent_jet.v * math.sin(opponent_jet.theta) * time_to_target) % config.arena_height

    dx, dy = toroidal_delta(ego_jet.x, ego_jet.y, lead_x, lead_y, config.arena_width, config.arena_height)
    diff = relative_bearing(dx, dy, ego_jet.theta)

    turn = max(-1.0, min(1.0, diff / ego_jet.w_max))
    fire = 1.0 if abs(diff) <= math.radians(config.fire_cone_angle_deg) else 0.0

    return np.array([turn, config.rule_based_throttle, fire], dtype=np.float32)


def _jet_rng(jet):
    """A per-jet RNG seeded from the jet's spawn pose.

    State lives on the jet (fresh each episode), so behavior is independent
    across episodes and parallel envs yet reproducible under a fixed env seed
    (spawns are drawn from the env's seeded RNG). Avoids global RNG state.
    """
    if not hasattr(jet, "_policy_rng"):
        seed = int(abs(jet.x * 1000.0 + jet.y * 7.0 + jet.theta * 13.0)) & 0xFFFFFFFF
        jet._policy_rng = np.random.default_rng(seed)
    return jet._policy_rng


def evasive_policy(ego_jet, opponent_jet, config, flip_every=30):
    """Weave to keep the attacker roughly perpendicular; full throttle, no fire.

    Randomizes which side (±90°) it drives the attacker toward every
    ``flip_every`` steps. Tests whether the agent can chase and finish a runner.
    """
    rng = _jet_rng(ego_jet)
    step = getattr(ego_jet, "_evasive_step", 0)
    if step % flip_every == 0:
        ego_jet._evasive_dir = 1.0 if rng.random() < 0.5 else -1.0
    ego_jet._evasive_step = step + 1

    dx, dy = toroidal_delta(
        ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, config.arena_width, config.arena_height
    )
    bearing_to_attacker = relative_bearing(dx, dy, ego_jet.theta)
    target_bearing = ego_jet._evasive_dir * (math.pi / 2)

    turn = max(-1.0, min(1.0, wrap_angle(bearing_to_attacker - target_bearing) / ego_jet.w_max))
    return np.array([turn, 1.0, 0.0], dtype=np.float32)


def random_policy(ego_jet, opponent_jet, config):
    """Uniform-random actions with occasional fire — a performance floor."""
    rng = _jet_rng(ego_jet)
    return np.array(
        [rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0), 1.0 if rng.random() < 0.1 else 0.0],
        dtype=np.float32,
    )


SCRIPTED_OPPONENTS = {
    "pure_pursuit": pure_pursuit_policy,
    "lead_pursuit": lead_pursuit_policy,
    "evasive": evasive_policy,
    "random": random_policy,
}
