import math

import numpy as np

from envs.physics import relative_bearing, toroidal_delta


def pure_pursuit_policy(ego_jet, opponent_jet, config):
    dx, dy = toroidal_delta(
        ego_jet.x, ego_jet.y, opponent_jet.x, opponent_jet.y, config.arena_width, config.arena_height
    )
    diff = relative_bearing(dx, dy, ego_jet.theta)

    turn = diff / ego_jet.w_max
    turn = max(-1.0, min(1.0, turn))

    fire = 1.0 if abs(diff) <= math.radians(config.fire_cone_angle_deg) else 0.0

    return np.array([turn, config.rule_based_throttle, fire], dtype=np.float32)
