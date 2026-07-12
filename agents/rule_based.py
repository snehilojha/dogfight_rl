import math

import numpy as np

from envs.reward import toroidal_relative_position


def angle_diff(target_angle, current_angle):
    diff = target_angle - current_angle
    return (diff + math.pi) % (2 * math.pi) - math.pi


def pure_pursuit_policy(ego_jet, opponent_jet, config):
    dx, dy = toroidal_relative_position(ego_jet, opponent_jet, config)
    target_angle = math.atan2(dy, dx)
    diff = angle_diff(target_angle, ego_jet.theta)

    turn = diff / ego_jet.w_max
    turn = max(-1.0, min(1.0, turn))

    fire = 1.0 if abs(diff) <= math.radians(config.fire_cone_angle_deg) else 0.0

    return np.array([turn, config.rule_based_throttle, fire], dtype=np.float32)
