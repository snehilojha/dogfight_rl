import math

import numpy as np

from envs.physics import relative_bearing, toroidal_delta


def build_obs(ego_jet, opponent_jet, config) -> np.ndarray:
    """
    Build observation vector for PPO agent.

    Args:
        ego_jet: Ego jet object with attributes: x, y, theta, v, gun_cooldown, health
        opponent_jet: Opponent jet object with attributes: x, y, theta
        config: DogfightConfig instance

    Returns:
        np.ndarray: 14-dimensional observation vector
    """
    arena_w = config.arena_width
    arena_h = config.arena_height

    obs = np.zeros(14, dtype=np.float32)
    obs[0] = ego_jet.x / arena_w * 2 - 1
    obs[1] = ego_jet.y / arena_h * 2 - 1
    obs[2] = math.sin(ego_jet.theta)
    obs[3] = math.cos(ego_jet.theta)
    obs[4] = ego_jet.v / config.max_speed

    # === OPPONENT RELATIVE STATE ===
    dx, dy = toroidal_delta(ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, arena_w, arena_h)
    obs[5] = dx / arena_w
    obs[6] = dy / arena_h

    distance = math.sqrt(dx * dx + dy * dy)
    obs[7] = distance / config.arena_diag

    rel_angle = relative_bearing(dx, dy, ego_jet.theta)

    obs[8] = math.sin(rel_angle)
    obs[9] = math.cos(rel_angle)

    obs[10] = math.sin(opponent_jet.theta)
    obs[11] = math.cos(opponent_jet.theta)

    # === TACTICAL STATE ===
    obs[12] = ego_jet.gun_cooldown / config.max_cooldown
    obs[13] = ego_jet.health / config.max_health
    return obs
